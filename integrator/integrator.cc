/**
  * @file integrator.cc
  * @brief Main application file for accelerated integration processing
  *        containing the necessary data loading, argument parsing, 
  *        threading, data preparation, and GPU kernel invocation in
  *        order to perform GPU-accelerated integration processing.
 */

#include "integrator.cuh"

#include <bitshuffle.h>

#include <Eigen/Dense>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <csignal>
#include <dx2/beam.hpp>
#include <dx2/beam_ops.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>
#include <gemmi/unitcell.hpp>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stop_token>
#include <string>
#include <thread>

#ifdef __linux__
#include <sched.h>
#endif

#include "../spotfinder/cbfread.hpp"
#include "../spotfinder/shmread.hpp"
#include "common.hpp"
#include "cuda_arg_parser.hpp"
#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"
#include "integrator.cuh"
#include "integrator/background.cuh"
#include "integrator/background.hpp"
#include "integrator/coordinate_system.hpp"
#include "integrator/extent.hpp"
#include "integrator/lp_correction.hpp"
#include "integrator/sigma_estimation.hpp"
#include "kabsch.cuh"
#include "math/device_precision.cuh"
#include "math/math_utils.cuh"
#include "math/vector3d.cuh"
#include "predictor/predict.hpp"
#include "version.hpp"

using Eigen::Vector3d;

constexpr size_t IntegratedSum = (1 << 8);

static FGAlgorithm parse_fg_algorithm(const std::string &name) {
    if (name == "ellipsoid") return FGAlgorithm::Ellipsoid;
    if (name == "dials") return FGAlgorithm::Dials;
    throw std::runtime_error("Unknown foreground algorithm: " + name);
}

static BackgroundModel parse_background_model(const std::string &name) {
    if (name == "constant" || name == "tukey") return BackgroundModel::Constant;
    if (name == "glm") return BackgroundModel::Glm;
    throw std::runtime_error("Unknown background model: " + name);
}

using namespace std::chrono_literals;

// Define a 2D mdspan type alias for convenience
using mdspan_2d =
  std::experimental::mdspan<scalar_t, std::experimental::dextents<size_t, 2>>;

// Conversion helpers for Eigen to CUDA vector types
/**
 * @brief Convert Eigen::Vector3d to fastvec::Vector3D
 */
inline fastvec::Vector3D to_vector3d(const Eigen::Vector3d &v) {
    return fastvec::make_vector3d(static_cast<scalar_t>(v.x()),
                                  static_cast<scalar_t>(v.y()),
                                  static_cast<scalar_t>(v.z()));
}

/**
 * @brief Convert mdspan row to fastvec::Vector3D
 */
template <typename MdspanType>
inline fastvec::Vector3D to_vector3d(const MdspanType &mdspan, size_t row) {
    return fastvec::make_vector3d(static_cast<scalar_t>(mdspan(row, 0)),
                                  static_cast<scalar_t>(mdspan(row, 1)),
                                  static_cast<scalar_t>(mdspan(row, 2)));
}

// Global stop token for picking up user cancellation
std::stop_source global_stop;

// Function for passing to std::signal to register the stop request
extern "C" void stop_processing(int sig) {
    if (global_stop.stop_requested()) {
        // We already requested before, but we want it faster. Abort.
        std::quick_exit(1);
    } else {
        fmt::print("Running interrupted by user request\n");
        global_stop.request_stop();
    }
}

/**
 * @brief Determine a sensible default number of worker threads.
 *
 * Prefers the number of CPUs the process is actually allowed to run on
 * (respecting cgroup/scheduler affinity for when in containers).
 * Falls back to std::thread::hardware_concurrency(), and finally to 1.
 */
static uint32_t auto_select_thread_count() {
#ifdef __linux__
    cpu_set_t cpus;
    if (sched_getaffinity(0, sizeof(cpus), &cpus) == 0) {
        int count = CPU_COUNT(&cpus);
        logger.debug("sched_getaffinity reports {} allowed CPU(s)", count);
        if (count > 0) {
            return static_cast<uint32_t>(count);
        }
    } else {
        logger.debug("sched_getaffinity failed, falling back to hardware_concurrency");
    }
#endif
    uint32_t hw_threads = std::thread::hardware_concurrency();
    logger.debug("std::thread::hardware_concurrency() reports {} thread(s)",
                 hw_threads);
    return hw_threads ? hw_threads : 1;
}

#pragma region Argument Parsing
class IntegratorArgumentParser : public CUDAArgumentParser {
  public:
    IntegratorArgumentParser(std::string version) : CUDAArgumentParser(version) {
        add_h5read_arguments();      // Override to use refl + expt
        add_integrator_arguments();  // Add integrator-specific args
    }

    void add_h5read_arguments() override {
        add_argument("--reflection", "-r")
          .metavar("strong.refl")
          .help("Input reflection table")
          .action([&](const std::string &value) { _reflection_filepath = value; });

        add_argument("--experiment", "-e")
          .metavar("experiments.expt")
          .help("Input experiment list")
          .action([&](const std::string &value) { _experiment_filepath = value; });

        add_argument("--images", "-i")
          .metavar("images.nxs")
          .help("Input images file")
          .action([&](const std::string &value) { _images_filepath = value; });

        _activated_h5read = true;
    }

    auto const reflections() const -> std::string {
        return _reflection_filepath;
    }
    auto const experiment() const -> std::string {
        return _experiment_filepath;
    }
    auto const images() const -> std::string {
        return _images_filepath;
    }

  private:
    std::string _reflection_filepath;
    std::string _experiment_filepath;
    std::string _images_filepath;

    void add_integrator_arguments() {
        add_argument("-n", "--threads")
          .help("Number of parallel reader threads (default: 0 = auto-select)")
          .default_value<uint32_t>(0)
          .metavar("NUM")
          .scan<'u', uint32_t>();

        add_argument("--timeout")
          .help("Amount of time (in seconds) to wait for new images before failing.")
          .metavar("S")
          .default_value<float>(30.0f)
          .scan<'f', float>();

        add_argument("--sigma_m", "-sm")
          .help("Sigma_m: Standard deviation of the rotation axis in reciprocal space.")
          .metavar("σm")
          .scan<'f', float>();

        add_argument("--sigma_b", "-sb")
          .help(
            "Sigma_b: Standard deviation of the beam direction in reciprocal space.")
          .metavar("σb")
          .scan<'f', float>();

        add_argument("--sigma_estimation.min_bbox_depth", "--min_bbox_depth")
          .help(
            "When calculating sigma_m, only use reflections that span at least this "
            "number of images.")
          .default_value<int>(6)
          .scan<'i', int>();

        add_argument("-a", "--algorithm")
          .help("Foreground algorithm choice - dials or ellipsoid.")
          .default_value<std::string>("ellipsoid");

        add_argument("--background")
          .help(
            "Background model - constant (Tukey/IQR outlier rejection) or "
            "glm (robust-Poisson GLM constant background).")
          .default_value<std::string>("constant");

        add_argument("--min_zeta")
          .help(
            "Reflections close to the rotation axis, with a zeta below this threshold "
            "will not be integrated.")
          .default_value<float>(0.05f)
          .scan<'f', float>();

        add_argument("--output")
          .help("Output file path")
          .metavar("integrated.refl")
          .default_value<std::string>("integrated.refl");
    }
};
#pragma endregion Argument Parsing

#pragma region Application Entry
int main(int argc, char **argv) {
    logger.info("Version: {}", FFS_VERSION);

    // Parse arguments
    auto parser = IntegratorArgumentParser(FFS_VERSION);
    auto args = parser.parse_args(argc, argv);
    const auto reflection_file = parser.reflections();
    const auto experiment_file = parser.experiment();
    float wait_timeout = parser.get<float>("timeout");

    // Guard against missing files
    if (!std::filesystem::exists(reflection_file)) {
        logger.error("Reflection file not found: {}", reflection_file);
        return 1;
    }
    if (!std::filesystem::exists(experiment_file)) {
        logger.error("Experiment file not found: {}", experiment_file);
        return 1;
    }

    // Load reflection data
    logger.info("Loading data from file: {}", reflection_file);
    ReflectionTable reflections(reflection_file);

#pragma region Data preparation
    // Parse experiment list from JSON
    std::ifstream f(experiment_file);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error &ex) {
        logger.error("Failed to parse experiment file '{}': byte {}, {}",
                     experiment_file,
                     ex.byte,
                     ex.what());
        return 1;
    }

    // Construct Experiment object and extract components
    Experiment expt;
    try {
        expt = Experiment(elist_json_obj);
    } catch (const std::invalid_argument &ex) {
        logger.error(
          "Failed to construct Experiment from '{}': {}", experiment_file, ex.what());
        return 1;
    }

    // Extract experimental components
    auto &beam = beam_ops::require_monochromatic(expt.beam());
    Goniometer gonio = expt.goniometer();
    const Panel &panel = expt.detector().panels()[0];  // Assuming single panel detector
    const Scan &scan = expt.scan();
    const Crystal &crystal = expt.crystal();

    // Extract vectors and parameters
    Eigen::Vector3d s0 = beam.get_s0();
    Eigen::Vector3d rotation_axis = gonio.get_rotation_axis();
    const auto [osc_start, osc_width] = scan.get_oscillation();
    int image_range_start = scan.get_image_range()[0];
    int image_range_end = scan.get_image_range()[1];
    double wl = beam.get_wavelength();
    gemmi::UnitCell cell = crystal.get_unit_cell();

    // Foreground algorithm and zeta cutoff for host-side filtering.
    FGAlgorithm fg_algorithm = parse_fg_algorithm(parser.get<std::string>("algorithm"));
    float min_zeta = parser.get<float>("min_zeta");
    std::string output_file = parser.get<std::string>("output");
    logger.info("Foreground algorithm: {}",
                fg_algorithm == FGAlgorithm::Ellipsoid ? "ellipsoid" : "dials");

    BackgroundModel background_model =
      parse_background_model(parser.get<std::string>("background"));
    logger.info("Background model: {}", parser.get<std::string>("background"));

#pragma endregion Data preparation

#pragma region Sigma estimation
    // If input is a predicted refl, then we require sigma_b, sigma_m as we will not be
    // able to estimate it from the data
    // Else the input as an indexed.refl/refined.refl with the sigma variance columns,
    // then we can calculate sigma_b, sigma_m but will have to also run the predict
    // code in this program.
    float sigma_b = 0.0;
    float sigma_m = 0.0;
    if (parser.is_used("sigma_m")) {
        sigma_m = degrees_to_radians(
          parser.get<float>("sigma_m"));  // Use radians for calculations
    }
    if (parser.is_used("sigma_b")) {
        sigma_b = degrees_to_radians(
          parser.get<float>("sigma_b"));  // Use radians for calculations
    }

    // Estimate sigmas
    auto sigma_b_data = reflections.column<double>("sigma_b_variance");
    auto sigma_m_data = reflections.column<double>("sigma_m_variance");
    auto extent_z_data = reflections.column<int>("spot_extent_z");
    if (sigma_b_data && sigma_m_data && extent_z_data) {
        int min_bbox_depth = parser.get<int>("sigma_estimation.min_bbox_depth");
        // Estimate the values from the data, and use if user hasn't specified values.
        auto [sigma_b_calc, sigma_m_calc] =
          estimate_sigmas(reflections, expt, min_bbox_depth);
        if (sigma_m == 0.0) {
            sigma_m = sigma_m_calc;
        }
        if (sigma_b == 0.0) {
            sigma_b = sigma_b_calc;
        }
    }
    if (sigma_b == 0.0) {
        logger.error(
          "No value for sigma_b. This must either be provided as input, or an input "
          "reflection "
          "table containing sigma_b_variance must be used.");
        return 1;
    }
    if (sigma_m == 0.0) {
        logger.error(
          "No value for sigma_m. This must either be provided as input, or an input "
          "reflection "
          "table containing sigma_m_variance and spot_extent_z must be used.");
        return 1;
    }
#pragma endregion Sigma estimation

#pragma region Predict or extract predictions
    // Determine if the data are predicted based on the reflection flags
    // If data are not predicted, run predict code.
    auto flags_column_opt = reflections.column<size_t>("flags");
    if (!flags_column_opt) {
        logger.error("Column 'flags' not found.");
        return 1;
    }
    auto flags = *flags_column_opt;
    bool all_predicted = true;
    for (int i = 0; i < flags.extent(0); ++i) {
        auto f = flags(i, 0);
        if (!(f & predicted_flag)) {
            all_predicted = false;
            break;
        }
    }
    if (all_predicted) {
        logger.info(
          "Input data have the predict flag set, treating as predicted data.");
    }

    mdspan_type<double> phi_column;
    mdspan_type<double> s1_vectors;
    std::vector<int> hkl_vectors;
    size_t num_reflections;
    predicted_data_rotation output_data;  // Define here so that members stay in scope
    if (!all_predicted) {
        scan_varying_data sv_data;
        bool scan_varying = false;
        std::tie(scan_varying, sv_data) =
          extract_scan_varying_data(elist_json_obj, scan);
        if (scan_varying) {
            logger.info("Monochromatic scan-varying prediction");
        } else {
            logger.info("Monochromatic static prediction");
        }

        double wavelength = beam.get_wavelength();
        double dmin_min = 0.5 * wavelength;
        // FIXME: Need a better dmin_default from .expt file (like in DIALS)
        double dmin_default = dmin_min;
        double param_dmin = dmin_default;
        size_t max_threads = std::thread::hardware_concurrency();
        size_t nthreads = max_threads ? max_threads : 1;
        int buffer_size = 0;
        output_data =
          predict_rotation(expt, sv_data, param_dmin, buffer_size, nthreads);
        std::size_t num_new_reflections = output_data.panels.size();
        logger.info("Predicted {} reflections", num_new_reflections);

        s1_vectors =
          mdspan_type<double>(output_data.s1.data(), output_data.s1.size() / 3, 3);
        phi_column = mdspan_type<double>(
          output_data.xyz_mm.data(), output_data.xyz_mm.size() / 3, 3);
        num_reflections = output_data.enter.size();
        hkl_vectors = output_data.hkl;
    } else {
        // is already predicted, so just extract required data.
        auto s1_vectors_opt = reflections.column<double>("s1");
        if (!s1_vectors_opt) {
            logger.error("Column 's1' not found in reflection data.");
            return 1;
        }
        s1_vectors = *s1_vectors_opt;
        auto phi_column_opt = reflections.column<double>("xyzcal.mm");
        if (!phi_column_opt) {
            logger.error("Column 'xyzcal.mm' not found for phi positions.");
            return 1;
        }
        phi_column = *phi_column_opt;
        auto hkl_opt = reflections.column<int>("miller_index");
        if (!hkl_opt) {
            logger.error("Column 'miller_index' not found in reflection data.");
            return 1;
        }
        num_reflections = s1_vectors.extent(0);
        hkl_vectors =
          std::vector<int>(hkl_opt.value().data_handle(),
                           hkl_opt.value().data_handle() + hkl_opt.value().size());
    }

#pragma endregion Predict or extract predictions

#pragma region Image Reading and Threading
    // Now set up for multi-threaded image reading and processing
    logger.info("Setting up image reading and threading");

    std::signal(SIGINT, stop_processing);

    // Compute bounding boxes using CPU-based Kabsch coordinate system
    logger.info(
      "Computing Kabsch bounding boxes for {} reflections using CPU implementation",
      num_reflections);

    // Call the CPU-based extent function
    // Note: compute_kabsch_bounding_boxes expects double precision mdspan,
    // so we pass the original double precision s1_vectors and phi_column
    std::vector<BoundingBoxExtents> computed_bboxes =
      compute_kabsch_bounding_boxes(s0,
                                    rotation_axis,
                                    s1_vectors,
                                    phi_column,
                                    num_reflections,
                                    sigma_b,
                                    sigma_m,
                                    panel,
                                    scan,
                                    beam);

    logger.info("Bounding box computation completed");

    // Convert BoundingBoxExtents to flat array format for storage
    std::vector<double> computed_bbox_data(num_reflections * 6);
    for (size_t i = 0; i < num_reflections; ++i) {
        const int step = 6 * i;
        computed_bbox_data[step + 0] = computed_bboxes[i].x_min;
        computed_bbox_data[step + 1] = computed_bboxes[i].x_max;
        computed_bbox_data[step + 2] = computed_bboxes[i].y_min;
        computed_bbox_data[step + 3] = computed_bboxes[i].y_max;
        computed_bbox_data[step + 4] = static_cast<double>(computed_bboxes[i].z_min);
        computed_bbox_data[step + 5] = static_cast<double>(computed_bboxes[i].z_max);
    }

    // Build per-reflection coordinate systems and apply host-side min_zeta
    // filter. Reflections with |zeta| < min_zeta are skipped (dont_integrate).
    std::vector<CoordinateSystem> coord_system_vector;
    coord_system_vector.reserve(num_reflections);
    std::vector<uint8_t> dont_integrate(num_reflections, 0);
    {
        size_t n_skipped = 0;
        for (size_t i = 0; i < num_reflections; ++i) {
            Vector3d s1_this(s1_vectors(i, 0), s1_vectors(i, 1), s1_vectors(i, 2));
            CoordinateSystem cs(rotation_axis, s0, s1_this, phi_column(i, 2));
            coord_system_vector.push_back(cs);
            if (std::abs(cs.zeta()) < min_zeta) {
                dont_integrate[i] = 1;
                ++n_skipped;
            }
        }
        logger.info("min_zeta={}: skipping {} of {} reflections",
                    min_zeta,
                    n_skipped,
                    num_reflections);
    }

    // Map reflections by z layer (image number), excluding dont_integrate ones
    logger.info("Mapping reflections by image number (z layer)");
    std::unordered_map<int, std::vector<size_t>> reflections_by_image;

    for (size_t refl_id = 0; refl_id < num_reflections; ++refl_id) {
        if (dont_integrate[refl_id]) continue;
        const auto &bbox = computed_bboxes[refl_id];

        // Add this reflection to all images it spans
        for (int z = bbox.z_min; z <= bbox.z_max; ++z) {
            reflections_by_image[z].push_back(refl_id);
        }
    }

    logger.info("Reflections mapped across {} unique images",
                reflections_by_image.size());

    // Log some statistics about the mapping
    if (!reflections_by_image.empty()) {
        size_t min_refls_per_image = std::numeric_limits<size_t>::max();
        size_t max_refls_per_image = 0;
        size_t total_refls = 0;

        for (const auto &[image, refls] : reflections_by_image) {
            min_refls_per_image = std::min(min_refls_per_image, refls.size());
            max_refls_per_image = std::max(max_refls_per_image, refls.size());
            total_refls += refls.size();
        }

        double avg_refls_per_image =
          static_cast<double>(total_refls) / reflections_by_image.size();
        logger.info("Reflections per image: min={}, max={}, avg={:.1f}",
                    min_refls_per_image,
                    max_refls_per_image,
                    avg_refls_per_image);
    }

    // Get threading parameters (0 = auto-select)
    uint32_t num_cpu_threads = parser.get<uint32_t>("threads");
    if (num_cpu_threads == 0) {
        num_cpu_threads = auto_select_thread_count();
        logger.info("Auto-selected {} CPU threads", num_cpu_threads);
    }
    logger.info("Running with {} CPU threads", num_cpu_threads);

    // Set up image reader. When no images are given, fall back to the
    // image file recorded in the experiment
    auto images_file = parser.images();
    if (images_file.empty()) {
        images_file = expt.imagesequence().filename();
        if (!images_file.empty()) {
            logger.info("No --images given; using image file from the experiment: {}",
                        images_file);
        }
    }
    std::unique_ptr<Reader> reader_ptr;

    if (std::filesystem::is_directory(images_file)) {
        reader_ptr = std::make_unique<SHMRead>(images_file);
    } else if (images_file.ends_with(".cbf")) {
        logger.error("CBF reading not yet supported in integrator mode");
        return 1;
    } else {
        reader_ptr = images_file.empty() ? std::make_unique<H5Read>()
                                         : std::make_unique<H5Read>(images_file);
    }

    Reader &reader = *reader_ptr;
    auto reader_mutex = std::mutex{};

    uint32_t num_images = reader.get_number_of_images();
    uint32_t height = reader.image_shape()[0];
    uint32_t width = reader.image_shape()[1];

    logger.info("Image dimensions: {} x {} = {} pixels", width, height, width * height);
    logger.info("Number of images: {}", num_images);

    auto all_images_start_time = std::chrono::high_resolution_clock::now();
    auto next_image = std::atomic<int>(0);
    auto completed_images = std::atomic<int>(0);
    auto cpu_sync = std::barrier{num_cpu_threads};

    double time_waiting_for_images = 0.0;

#pragma region Prep GPU Data Buffers
    // Get detector d_matrix and flatten for GPU
    Eigen::Matrix3d d_matrix_eigen = panel.get_d_matrix();
    std::vector<scalar_t> d_matrix_flat(9);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            d_matrix_flat[i * 3 + j] = static_cast<scalar_t>(d_matrix_eigen(i, j));
        }
    }

    // Convert s1_vectors to Vector3D array for GPU
    std::vector<fastvec::Vector3D> s1_vectors_vec(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        s1_vectors_vec[i] = to_vector3d(s1_vectors, i);
    }

    // phi_positions_converted_data already contains phi values (column index 2 of xyzcal.mm)
    // Need to extract just the phi component (3rd column)
    std::vector<scalar_t> phi_values_vec(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        phi_values_vec[i] = static_cast<scalar_t>(phi_column(i, 2));
    }

    // Allocate and copy GPU buffers (shared across all threads)
    DeviceBuffer<scalar_t> d_d_matrix(9);
    DeviceBuffer<fastvec::Vector3D> d_s1_vectors(num_reflections);
    DeviceBuffer<scalar_t> d_phi_values(num_reflections);
    DeviceBuffer<BoundingBoxExtents> d_bboxes(num_reflections);

    d_d_matrix.assign(d_matrix_flat.data());
    d_s1_vectors.assign(s1_vectors_vec.data());
    d_phi_values.assign(phi_values_vec.data());
    d_bboxes.assign(computed_bboxes.data());

    // Upload the detector mask once (constant across all images).
    // non-zero = valid, 0 = masked. If the reader provides no mask,
    // treat every pixel as valid. Stored flat (width*height), indexed
    // `gy * width + gx`.
    std::vector<uint8_t> h_mask(static_cast<size_t>(width) * height, 1);
    if (auto mask_opt = reader.get_mask()) {
        const auto &mask_span = *mask_opt;
        std::copy(mask_span.begin(), mask_span.end(), h_mask.begin());
        logger.info("Using detector mask from reader ({} pixels)", mask_span.size());
    } else {
        logger.info("No detector mask available; treating all pixels as valid");
    }
    DeviceBuffer<uint8_t> d_mask(static_cast<size_t>(width) * height);
    d_mask.assign(h_mask.data());

    // Reflection bboxes are not clamped to the detector in x/y, so
    // foreground pixels can fall outside the image. The launch grid is
    // padded to span this overflow; its origin may be negative.
    // Foreground pixels landing outside the image fail their reflection
    // in the kernel.
    int grid_origin_x = 0;
    int grid_origin_y = 0;
    int grid_max_x = static_cast<int>(width);
    int grid_max_y = static_cast<int>(height);
    for (size_t i = 0; i < num_reflections; ++i) {
        if (dont_integrate[i]) continue;
        const auto &bb = computed_bboxes[i];
        grid_origin_x = std::min(grid_origin_x, bb.x_min);
        grid_origin_y = std::min(grid_origin_y, bb.y_min);
        grid_max_x = std::max(grid_max_x, bb.x_max + 1);
        grid_max_y = std::max(grid_max_y, bb.y_max + 1);
    }
    uint32_t grid_w = static_cast<uint32_t>(grid_max_x - grid_origin_x);
    uint32_t grid_h = static_cast<uint32_t>(grid_max_y - grid_origin_y);
    logger.info("Kabsch launch grid: origin=({},{}), extent={}x{}",
                grid_origin_x,
                grid_origin_y,
                grid_w,
                grid_h);

    DetectorParameters det_params = make_detector_params(panel);

    // Convert beam parameters to Vector3D
    fastvec::Vector3D s0_vec = to_vector3d(s0);
    fastvec::Vector3D rot_axis_vec = to_vector3d(rotation_axis);
    scalar_t wavelength = static_cast<scalar_t>(wl);
    scalar_t osc_start_scalar = static_cast<scalar_t>(osc_start);
    scalar_t osc_width_scalar = static_cast<scalar_t>(osc_width);

    // Summation integration parameters
    // delta_b and delta_m define the foreground ellipsoid extent in Kabsch space
    constexpr scalar_t n_sigma = 3.0;  // Default number of sigma for foreground cutoff
    const scalar_t delta_b = n_sigma * static_cast<scalar_t>(sigma_b);
    const scalar_t delta_m = n_sigma * static_cast<scalar_t>(sigma_m);
    logger.info("Summation integration: delta_b={:.6f}, delta_m={:.6f} (n_sigma={})",
                delta_b,
                delta_m,
                n_sigma);

    // Allocate accumulator buffers for summation integration (shared across all threads)
    // These persist across all images and accumulate atomically
    DeviceBuffer<accumulator_t> d_foreground_sum(num_reflections);
    DeviceBuffer<uint32_t> d_foreground_count(num_reflections);
    // Per-reflection background histogram: one bin per integer pixel value over
    // [0, NUM_BG_BINS), plus an overflow counter for the high tail. The Kabsch
    // kernel fills these; compute_background() reduces them into an estimate
    // after the image loop.
    DeviceBuffer<uint32_t> d_background_hist(num_reflections * NUM_BG_BINS);
    DeviceBuffer<uint32_t> d_background_overflow(num_reflections);
    logger.info("Background histograms: {} reflections x {} bins = {:.1f} MiB",
                num_reflections,
                NUM_BG_BINS,
                (num_reflections * NUM_BG_BINS * sizeof(uint32_t)) / (1024.0 * 1024.0));
    // Reduced background estimates (written by compute_background()).
    DeviceBuffer<double> d_background_mean(num_reflections);
    DeviceBuffer<double> d_background_sum_value(num_reflections);
    DeviceBuffer<uint32_t> d_background_count(num_reflections);
    DeviceBuffer<uint8_t> d_background_success(num_reflections);
    // COM (intensity * coord) accumulators. Baseline uses Vector3d (double) for
    // these, but we use unsigned long long so we can use the hardware-fast
    // integer atomicAdd; double atomicAdd is slower and requires sm_60+. To
    // keep arithmetic integer we accumulate intensity * (2*coord + 1) instead
    // of intensity * (coord + 0.5), then divide by 2*intensity in finalisation
    // (algebraically identical). 64-bit gives plenty of headroom against
    // overflow for realistic pixel values and reflection sizes.
    DeviceBuffer<unsigned long long> d_intensity_times_x(num_reflections);
    DeviceBuffer<unsigned long long> d_intensity_times_y(num_reflections);
    DeviceBuffer<unsigned long long> d_intensity_times_z(num_reflections);
    // Per-reflection success flag (1 = ok). Zeroed by the kernel when a
    // foreground pixel is masked or falls outside the image, and on the
    // host after the loop if fg_count == 0.
    DeviceBuffer<uint8_t> d_success(num_reflections);

    cudaMemset(d_foreground_sum.data(), 0, num_reflections * sizeof(accumulator_t));
    cudaMemset(d_foreground_count.data(), 0, num_reflections * sizeof(uint32_t));
    cudaMemset(
      d_background_hist.data(), 0, num_reflections * NUM_BG_BINS * sizeof(uint32_t));
    cudaMemset(d_background_overflow.data(), 0, num_reflections * sizeof(uint32_t));
    cudaMemset(
      d_intensity_times_x.data(), 0, num_reflections * sizeof(unsigned long long));
    cudaMemset(
      d_intensity_times_y.data(), 0, num_reflections * sizeof(unsigned long long));
    cudaMemset(
      d_intensity_times_z.data(), 0, num_reflections * sizeof(unsigned long long));
    // Pre-fill success with dont_integrate complement (1 = ok, 0 = skipped/failed).
    std::vector<uint8_t> success_init(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        success_init[i] = dont_integrate[i] ? 0 : 1;
    }
    d_success.assign(success_init.data());
    cuda_throw_error();
#pragma endregion Prep GPU Data Buffers

#pragma region Thread launch
    logger.info("Starting image reading and processing threads");
    // Spawn the reader threads
    std::vector<std::jthread> threads;
    for (int thread_id = 0; thread_id < num_cpu_threads; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            auto stop_token = global_stop.get_token();
            CudaStream stream;  // Per-thread CUDA stream

            // Full image buffers for decompression (pinned memory for faster transfer)
            auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height);
            auto raw_chunk_buffer =
              std::vector<uint8_t>(width * height * sizeof(pixel_t));

            // Device memory for GPU processing
            auto device_image = PitchedMalloc<pixel_t>(width, height);

            // Let all threads do setup tasks before reading starts
            cpu_sync.arrive_and_wait();

            auto last_image_received = std::chrono::high_resolution_clock::now();

            while (!stop_token.stop_requested()) {
                auto image_num = next_image.fetch_add(1);
                if (image_num >= num_images) {
                    break;
                }

                // Check if this image has any reflections
                if (reflections_by_image.find(image_num)
                    == reflections_by_image.end()) {
                    completed_images += 1;
                    continue;
                }

                {
                    std::scoped_lock lock(reader_mutex);
                    auto swmr_wait_start_time =
                      std::chrono::high_resolution_clock::now();

                    // Check that our image is available and wait if not
                    while (!reader.is_image_available(image_num)
                           && !stop_token.stop_requested()) {
                        auto current_time = std::chrono::high_resolution_clock::now();
                        auto elapsed_wait_time =
                          std::chrono::duration_cast<std::chrono::duration<double>>(
                            current_time - last_image_received)
                            .count();

                        if (elapsed_wait_time > wait_timeout) {
                            logger.error("Timeout waiting for image {}", image_num);
                            global_stop.request_stop();
                            break;
                        }

                        std::this_thread::sleep_for(100ms);
                    }

                    if (stop_token.stop_requested()) {
                        break;
                    }

                    last_image_received = std::chrono::high_resolution_clock::now();
                    time_waiting_for_images +=
                      std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now()
                        - swmr_wait_start_time)
                        .count();
                }

                // Fetch the image data from the reader
                std::span<uint8_t> buffer;
                while (true) {
                    {
                        std::scoped_lock lock(reader_mutex);
                        buffer = reader.get_raw_chunk(image_num, raw_chunk_buffer);
                    }

                    if (buffer.size() == 0) {
                        logger.warn("Got buffer size 0 for image {}. Sleeping.",
                                    image_num);
                        std::this_thread::sleep_for(100ms);
                        continue;
                    }
                    break;
                }

                // Decompress the data into pinned host memory
                switch (reader.get_raw_chunk_compression()) {
                case Reader::ChunkCompression::BITSHUFFLE_LZ4:
                    bshuf_decompress_lz4(buffer.data() + 12,
                                         host_image.get(),
                                         width * height,
                                         sizeof(pixel_t),
                                         0);
                    break;
                case Reader::ChunkCompression::BYTE_OFFSET_32:
                    decompress_byte_offset<pixel_t>(
                      buffer,
                      {host_image.get(),
                       static_cast<std::span<pixel_t>::size_type>(width * height)});
                    break;
                }

                // Copy image data to device memory
                cudaMemcpy2DAsync(device_image.get(),
                                  device_image.pitch_bytes(),
                                  host_image.get(),
                                  width * sizeof(pixel_t),
                                  width * sizeof(pixel_t),
                                  height,
                                  cudaMemcpyHostToDevice,
                                  stream);
                cuda_throw_error();

                // Get reflection indices for this image and copy to device
                const auto &refl_indices = reflections_by_image[image_num];
                size_t num_refls_this_image = refl_indices.size();

                // Allocate per-image device buffer for reflection indices
                DeviceBuffer<size_t> d_reflection_indices(num_refls_this_image);
                d_reflection_indices.assign(refl_indices.data());

                // Launch Kabsch transform kernel for this image
                // This computes Kabsch coordinates and atomically accumulates
                // foreground/background intensities for summation integration
                logger.info("Launching GPU kernel for image {} with {} reflections",
                            image_num,
                            num_refls_this_image);
                compute_kabsch_transform(device_image.get(),
                                         device_image.pitch_bytes(),
                                         width,
                                         height,
                                         image_num,
                                         d_d_matrix.data(),
                                         wavelength,
                                         det_params,
                                         osc_start_scalar,
                                         osc_width_scalar,
                                         image_range_start,
                                         s0_vec,
                                         rot_axis_vec,
                                         d_s1_vectors.data(),
                                         d_phi_values.data(),
                                         d_bboxes.data(),
                                         d_reflection_indices.data(),
                                         num_refls_this_image,
                                         d_mask.data(),
                                         grid_origin_x,
                                         grid_origin_y,
                                         grid_w,
                                         grid_h,
                                         delta_b,
                                         delta_m,
                                         fg_algorithm,
                                         d_foreground_sum.data(),
                                         d_foreground_count.data(),
                                         d_background_hist.data(),
                                         d_background_overflow.data(),
                                         d_intensity_times_x.data(),
                                         d_intensity_times_y.data(),
                                         d_intensity_times_z.data(),
                                         d_success.data(),
                                         stream);

                logger.trace("Thread {} loaded image {}", thread_id, image_num);
                completed_images += 1;
            }
        });
    }

    // Wait for all threads to finish
    for (auto &thread : threads) {
        thread.join();
    }

    float total_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - all_images_start_time)
        .count();

    logger.info("{} images processed in {:.2f} s ({:.1f} fps)",
                int(completed_images),
                total_time,
                completed_images / total_time);

    if (time_waiting_for_images < 10) {
        logger.info("Total time waiting for images: {:.0f} ms",
                    time_waiting_for_images * 1000);
    } else {
        logger.info("Total time waiting for images: {:.2f} s", time_waiting_for_images);
    }

#pragma region Background Reduction
    // The reader threads have joined, which destroyed their per-thread streams,
    // but cudaStreamDestroy does not wait for queued work, so the accumulation
    // kernels may still be in flight. Synchronise the whole device before the
    // reduction reads the histograms those kernels filled.
    // Perhaps better to have each thread synchronize its stream before joining,
    // this is simper for now, but worth testing during profiling.
    cudaDeviceSynchronize();
    cuda_throw_error();

    // Reduce each reflection's background histogram into an estimate on the
    // device (Tukey/IQR constant model), then copy the small results back.
    logger.info("Reducing background histograms for {} reflections", num_reflections);
    compute_background(background_model,
                       d_background_hist.data(),
                       d_background_overflow.data(),
                       num_reflections,
                       d_background_mean.data(),
                       d_background_sum_value.data(),
                       d_background_count.data(),
                       d_background_success.data(),
                       0);
    cudaDeviceSynchronize();
    cuda_throw_error();
#pragma endregion Background Reduction

#pragma region Summation Integration Finalization
    // Host-side reduction and finalization
    // Copy accumulator buffers back from GPU and compute final intensities
    logger.info("Finalizing summation integration for {} reflections", num_reflections);

    // Allocate host buffers for results
    std::vector<accumulator_t> h_foreground_sum(num_reflections);
    std::vector<uint32_t> h_foreground_count(num_reflections);
    std::vector<double> h_background_mean(num_reflections);
    std::vector<double> h_background_sum_value(num_reflections);
    std::vector<uint32_t> h_background_count(num_reflections);
    std::vector<uint32_t> h_background_overflow(num_reflections);
    std::vector<uint8_t> h_background_success(num_reflections);
    std::vector<unsigned long long> h_intensity_times_x(num_reflections);
    std::vector<unsigned long long> h_intensity_times_y(num_reflections);
    std::vector<unsigned long long> h_intensity_times_z(num_reflections);
    std::vector<uint8_t> h_success(num_reflections);

    // Copy results from device to host
    d_foreground_sum.extract(h_foreground_sum.data());
    d_foreground_count.extract(h_foreground_count.data());
    d_background_mean.extract(h_background_mean.data());
    d_background_sum_value.extract(h_background_sum_value.data());
    d_background_count.extract(h_background_count.data());
    d_background_overflow.extract(h_background_overflow.data());
    d_background_success.extract(h_background_success.data());
    d_intensity_times_x.extract(h_intensity_times_x.data());
    d_intensity_times_y.extract(h_intensity_times_y.data());
    d_intensity_times_z.extract(h_intensity_times_z.data());
    d_success.extract(h_success.data());

    // The background histogram only covers pixel values [0, NUM_BG_BINS). If a
    // reflection pushes more than kBackgroundMaxOverflowFraction of its
    // background pixels into the overflow tail, that range is too small to
    // characterise its background and the device Tukey estimate diverges from a
    // full-range computation. Fail loudly so the histogram range is raised
    // rather than silently producing a degraded intensity.
    {
        size_t overflowing = 0;
        double worst_fraction = 0.0;
        for (size_t i = 0; i < num_reflections; ++i) {
            const uint32_t total = h_background_count[i];
            if (total == 0) continue;
            const double fraction = static_cast<double>(h_background_overflow[i])
                                    / static_cast<double>(total);
            if (fraction > kBackgroundMaxOverflowFraction) {
                overflowing++;
                worst_fraction = std::max(worst_fraction, fraction);
            }
        }
        if (overflowing > 0) {
            throw std::runtime_error(fmt::format(
              "{} reflection(s) put more than {:.0f}% of their background pixels "
              "above NUM_BG_BINS={} (worst {:.1f}%); the background histogram "
              "range is too small. Increase NUM_BG_BINS.",
              overflowing,
              kBackgroundMaxOverflowFraction * 100.0,
              NUM_BG_BINS,
              worst_fraction * 100.0));
        }
    }

    // Compute final intensities for each reflection
    std::vector<double> intensities(num_reflections);
    std::vector<double> variances(num_reflections);
    std::vector<double> sigmas(num_reflections);       // σ(I) = √Var(I)
    std::vector<double> backgrounds(num_reflections);  // b̄  (background mean per pixel)
    std::vector<double> background_sigmas(num_reflections);  // σ(b̄)
    size_t valid_reflections = 0;
    size_t background_failures = 0;  // fg present but Tukey estimate rejected

    for (size_t i = 0; i < num_reflections; ++i) {
        uint32_t fg_count = h_foreground_count[i];
        uint32_t bg_count = h_background_count[i];

        if (fg_count == 0) {
            // No foreground pixels - reflection not measured
            intensities[i] = 0.0;
            variances[i] = -1.0;  // Flag as invalid
            sigmas[i] = -1.0;
            backgrounds[i] = 0.0;
            background_sigmas[i] = -1.0;
            continue;
        }

        double fg_sum = h_foreground_sum[i];

        // Tally rejected estimates (no inlier pixels, or the IQR fence ran past
        // the histogram range) purely for error reporting and logging; the
        // reflection is marked unintegrated below regardless.
        if (!h_background_success[i]) {
            ++background_failures;
        }

        // Background mean b̄ from the device-side background reduction (the
        // selected model). When the estimate failed the model contributes nothing.
        double bg_mean = h_background_success[i] ? h_background_mean[i] : 0.0;

        // Subtract background:  I = Σcᵢ(fg) − n_fg · b̄
        double background_total = bg_mean * fg_count;
        double intensity = fg_sum - background_total;

        // Variance estimation
        //
        //   I = Σcᵢ(fg) - B,  where B = n_fg · b̄  (total background under foreground)
        //
        //   Var(I) = |I| + |B| · (1 + n_fg / n_bg)
        //
        // The |I| term is the Poisson variance of the net signal.
        // The |B| term accounts for the Poisson variance of the background,
        // with the (1 + n_fg/n_bg) factor propagating the uncertainty in
        // the background estimate from n_bg pixels to n_fg pixels.
        //
        double fg_bg_ratio =
          (bg_count > 0) ? static_cast<double>(fg_count) / bg_count : 0.0;
        double variance =
          std::abs(intensity) + std::abs(background_total) * (1.0 + fg_bg_ratio);
        double bg_variance = std::abs(background_total) * (1.0 + fg_bg_ratio);

        intensities[i] = intensity;
        variances[i] = variance;
        sigmas[i] = (variance > 0.0) ? std::sqrt(variance) : 0.0;
        backgrounds[i] = bg_mean;
        background_sigmas[i] = (bg_variance > 0.0) ? std::sqrt(bg_variance) : 0.0;
        valid_reflections++;
    }

    logger.info("Summation integration complete: {} valid reflections out of {}",
                valid_reflections,
                num_reflections);

    if (background_failures > 0) {
        logger.warn(
          "Background estimate rejected for {} of {} reflections with "
          "foreground pixels; NUM_BG_BINS may be too small for their "
          "background level",
          background_failures,
          num_reflections);
    }

    // Log some statistics
    if (valid_reflections > 0) {
        double sum_intensity = 0.0;
        double sum_sigma = 0.0;
        double sum_bg = 0.0;
        double max_intensity = -std::numeric_limits<double>::infinity();
        double min_intensity = std::numeric_limits<double>::infinity();
        size_t n_positive_isigma = 0;
        double sum_i_over_sigma = 0.0;
        for (size_t i = 0; i < num_reflections; ++i) {
            if (variances[i] >= 0) {
                sum_intensity += intensities[i];
                sum_sigma += sigmas[i];
                sum_bg += backgrounds[i];
                max_intensity = std::max(max_intensity, intensities[i]);
                min_intensity = std::min(min_intensity, intensities[i]);
                if (sigmas[i] > 0.0) {
                    sum_i_over_sigma += intensities[i] / sigmas[i];
                    n_positive_isigma++;
                }
            }
        }
        logger.info("Intensity statistics: min={:.1f}, max={:.1f}, mean={:.1f}",
                    min_intensity,
                    max_intensity,
                    sum_intensity / valid_reflections);
        logger.info("Mean σ(I)={:.2f}, mean background={:.2f}",
                    sum_sigma / valid_reflections,
                    sum_bg / valid_reflections);
        if (n_positive_isigma > 0) {
            logger.info("Mean I/σ(I)={:.2f} ({} reflections with σ>0)",
                        sum_i_over_sigma / n_positive_isigma,
                        n_positive_isigma);
        }
    }

#pragma endregion Summation Integration Finalization

#pragma region Output Reflection Table
    // Compute centre-of-mass (xyzobs.px.value), partiality, LP, and d-spacing,
    // and write an output ReflectionTable matching the baseline integrator.
    logger.info("Building output reflection table");

    std::vector<double> xyzobs_px(num_reflections * 3);
    std::vector<double> partialities(num_reflections);
    std::vector<double> lp_corrections(num_reflections);
    std::vector<double> d_values(num_reflections);
    std::vector<int> nfg_out(num_reflections);
    std::vector<int> nbg_out(num_reflections);
    std::vector<double> bg_sum_out(num_reflections);
    std::vector<double> bg_mean_out(num_reflections);

    Vector3d pn = beam.get_polarization_normal();
    double pf = beam.get_polarization_fraction();
    LPCorrection lpcalculator(s0, pn, pf, rotation_axis);

    std::vector<uint8_t> success_final(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        nfg_out[i] = static_cast<int>(h_foreground_count[i]);
        nbg_out[i] = static_cast<int>(h_background_count[i]);
        // background.sum.value is the modelled background summed over the
        // background pixels; background.mean is the per-pixel level. For the
        // constant model these are the Tukey inlier sum and mean; for the GLM
        // model the robust-Poisson sum (mean*N) and mean. Both come from the
        // device reduction.
        bg_sum_out[i] = h_background_success[i] ? h_background_sum_value[i] : 0.0;
        bg_mean_out[i] = h_background_success[i] ? h_background_mean[i] : 0.0;

        // Success: kernel-flagged AND fg_count > 0 AND a valid background
        // estimate AND not dont_integrated.
        bool ok = h_success[i] && (h_foreground_count[i] > 0) && h_background_success[i]
                  && !dont_integrate[i];
        success_final[i] = ok ? 1 : 0;

        // Centre-of-mass: kernel accumulated intensity·(2k+1), so divide by 2·I.
        double fg_sum_d = static_cast<double>(h_foreground_sum[i]);
        if (h_foreground_count[i] > 0 && fg_sum_d > 0.0) {
            xyzobs_px[i * 3 + 0] =
              static_cast<double>(h_intensity_times_x[i]) / (2.0 * fg_sum_d);
            xyzobs_px[i * 3 + 1] =
              static_cast<double>(h_intensity_times_y[i]) / (2.0 * fg_sum_d);
            xyzobs_px[i * 3 + 2] =
              static_cast<double>(h_intensity_times_z[i]) / (2.0 * fg_sum_d);
        } else {
            // Fallback to bbox centre (DIALS convention from centroid/simple/algorithm.h)
            const auto &bb = computed_bboxes[i];
            xyzobs_px[i * 3 + 0] = 0.5 * (bb.x_min + bb.x_max);
            xyzobs_px[i * 3 + 1] = 0.5 * (bb.y_min + bb.y_max);
            xyzobs_px[i * 3 + 2] = 0.5 * (bb.z_min + bb.z_max);
        }

        // Partiality from coordinate-system zeta integrated over bbox z-extent.
        double xyzcal_px_z = radians_to_degrees(phi_column(i, 2)) / osc_width;
        double phi = osc_start + ((xyzcal_px_z + 1 - image_range_start) * osc_width);
        double phia =
          osc_start + ((computed_bboxes[i].z_min + 1 - image_range_start) * osc_width);
        double phib =
          osc_start + ((computed_bboxes[i].z_max + 1 - image_range_start) * osc_width);
        double zeta = coord_system_vector[i].zeta();
        double c = std::abs(zeta) / (std::sqrt(2.0) * sigma_m);
        partialities[i] =
          0.5 * (std::erf(c * (phib - phi)) - std::erf(c * (phia - phi)));

        Vector3d s1_this(s1_vectors(i, 0), s1_vectors(i, 1), s1_vectors(i, 2));
        lp_corrections[i] = lpcalculator.calculate(s1_this);

        std::array<int, 3> hkl_this = {
          hkl_vectors[i * 3], hkl_vectors[i * 3 + 1], hkl_vectors[i * 3 + 2]};
        d_values[i] = cell.calculate_d(hkl_this);
    }

    // Build output reflection table.
    std::vector<std::string> identifiers = reflections.get_identifiers();
    std::vector<uint64_t> ids_in = reflections.get_experiment_ids();
    ReflectionTable integrated_data(ids_in, identifiers);

    std::vector<double> intensity_variance(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        intensity_variance[i] = (variances[i] < 0.0) ? 0.0 : variances[i];
    }

    std::vector<double> xyzcal_mm(phi_column.data_handle(),
                                  phi_column.data_handle() + phi_column.size());
    std::vector<double> s1_out(s1_vectors.data_handle(),
                               s1_vectors.data_handle() + s1_vectors.size());
    std::vector<int> id_col(num_reflections, 0);
    std::vector<std::size_t> final_flags(num_reflections, IntegratedSum);

    integrated_data.add_column("intensity.sum.value", num_reflections, 1, intensities);
    integrated_data.add_column(
      "intensity.sum.variance", num_reflections, 1, intensity_variance);
    integrated_data.add_column("partiality", num_reflections, 1, partialities);
    integrated_data.add_column("miller_index", num_reflections, 3, hkl_vectors);
    integrated_data.add_column("lp", num_reflections, 1, lp_corrections);
    integrated_data.add_column("d", num_reflections, 1, d_values);
    integrated_data.add_column("xyzcal.mm", num_reflections, 3, xyzcal_mm);
    integrated_data.add_column("xyzobs.px.value", num_reflections, 3, xyzobs_px);
    integrated_data.add_column("s1", num_reflections, 3, s1_out);
    integrated_data.add_column("id", num_reflections, 1, id_col);
    integrated_data.add_column("num_pixels.background", num_reflections, 1, nbg_out);
    integrated_data.add_column("num_pixels.foreground", num_reflections, 1, nfg_out);
    integrated_data.add_column("background.sum.value", num_reflections, 1, bg_sum_out);
    integrated_data.add_column("background.mean", num_reflections, 1, bg_mean_out);
    integrated_data.add_column("flags", num_reflections, 1, final_flags);

    std::vector<bool> success_bool(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i)
        success_bool[i] = success_final[i] != 0;
    ReflectionTable success_data = integrated_data.select(success_bool);
    int n_integrated =
      success_data.column<double>("intensity.sum.value").value().extent(0);
    logger.info("Writing {} integrated reflections to {}", n_integrated, output_file);
    success_data.write(output_file);
#pragma endregion Output Reflection Table

#pragma endregion Image Reading and Threading

    return 0;
}
