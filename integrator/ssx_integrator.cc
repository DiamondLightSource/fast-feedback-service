#include <bitshuffle.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
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

#include "../spotfinder/cbfread.hpp"
#include "../spotfinder/shmread.hpp"
#include "common.hpp"
#include "cuda_arg_parser.hpp"
#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"
#include "integrator/background.cuh"
#include "integrator/background.hpp"
#include "integrator/background_pool.hpp"
//#include "integrator/coordinate_system.hpp"
#include "integrator/extent.hpp"
//#include "integrator/lp_correction.hpp"
//#include "integrator/sigma_estimation.hpp"
#include "ellipsoid.cuh"
#include "math/device_precision.cuh"
#include "math/math_utils.cuh"
#include "math/vector3d.cuh"
//#include "predictor/predict.hpp"
#include "thread_count.hpp"
#include "version.hpp"
#include "ssx/ssx_integrate.hpp"
#include "ssx/mosaicity_parameterisation.hpp"
#include "predictor/predict.hpp"
#include "predictor/threadpool.hpp"

using Eigen::Vector3d;
using Eigen::Matrix3d;

// Define a 2D mdspan type alias for convenience
template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;

// Global stop token for picking up user cancellation
std::stop_source global_stop;

static BackgroundModel parse_background_model(const std::string &name) {
    if (name == "constant" || name == "tukey") return BackgroundModel::Constant;
    if (name == "glm") return BackgroundModel::Glm;
    throw std::runtime_error("Unknown background model: " + name);
}

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

inline fastvec::Vector3D to_vector3d(const Eigen::Vector3d &v) {
    return fastvec::make_vector3d(static_cast<scalar_t>(v.x()),
                                  static_cast<scalar_t>(v.y()),
                                  static_cast<scalar_t>(v.z()));
}

#pragma region Argument Parsing
class SSXIntegratorArgumentParser : public CUDAArgumentParser {
  public:
    SSXIntegratorArgumentParser(std::string version) : CUDAArgumentParser(version) {
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

        _activated_h5read = true;
    }

    auto const reflections() const -> std::string {
        return _reflection_filepath;
    }
    auto const experiment() const -> std::string {
        return _experiment_filepath;
    }

  private:
    std::string _reflection_filepath;
    std::string _experiment_filepath;

    void add_integrator_arguments() {
        add_argument("-n", "--threads")
          .help("Number of parallel reader threads (default: 0 = auto-select)")
          .default_value<uint32_t>(0)
          .metavar("NUM")
          .scan<'u', uint32_t>();
        add_argument("--background.pool_entries_per_reflection")
          .help(
            "Size of the shared background pool, in entries per reflection on "
            "average. Each entry holds one distinct background value of one "
            "reflection; reflections share the pool, so a single one may use many.")
          .metavar("N")
          .default_value(kBackgroundPoolEntriesPerReflection)
          .scan<'u', uint32_t>();
        add_argument("--background")
          .help(
            "Background model - constant (Tukey/IQR outlier rejection) or "
            "glm (robust-Poisson GLM constant background).")
          .default_value<std::string>("glm");
    }
};
#pragma endregion Argument Parsing

#pragma region Application Entry
int main(int argc, char **argv) {
    logger.info("Version: {}", FFS_VERSION);

    // Parse arguments
    auto parser = SSXIntegratorArgumentParser(FFS_VERSION);
    auto args = parser.parse_args(argc, argv);
    const auto reflection_file = parser.reflections();
    const auto experiment_file = parser.experiment();
    float wait_timeout = 1.0;

    // Get threading parameters (0 = auto-select)
    uint32_t num_cpu_threads = parser.get<uint32_t>("threads");
    if (num_cpu_threads == 0) {
        num_cpu_threads = auto_select_thread_count();
        logger.info("Auto-selected {} CPU threads", num_cpu_threads);
    }
    logger.info("Running with {} CPU threads", num_cpu_threads);

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
    Experiment expt;
    try {
        expt = Experiment(elist_json_obj);
    } catch (std::invalid_argument const &ex) {
        logger.error("Unable to create experiment: {}", ex.what());
        std::exit(1);
    }

    const uint32_t background_pool_entries_per_reflection =
      parser.get<uint32_t>("background.pool_entries_per_reflection");
    if (background_pool_entries_per_reflection == 0) {
        logger.error("--background.pool_entries_per_reflection must be at least 1");
        return 1;
    }

    BackgroundModel background_model =
      parse_background_model(parser.get<std::string>("background"));
    logger.info("Background model: {}", parser.get<std::string>("background"));

    auto &beam = beam_ops::require_monochromatic(expt.beam());
    Eigen::Vector3d s0 = beam.get_s0();
    double wavelength = beam.get_wavelength();
    const Panel &panel = expt.detector().panels()[0];  // Assuming single panel detector
    const Crystal &crystal = expt.crystal();
    Eigen::Matrix3d A = crystal.get_A_matrix();

    // want a map of experiment identifier to image number
    std::unordered_map<std::string, int> identifier_to_image_number;
    std::unordered_map<std::string, Crystal> identifier_to_crystal_model;
    auto scans = elist_json_obj["scan"];
    for (auto expt: elist_json_obj["experiment"]){
        std::string identifier = expt["identifier"];
        int crystal_id = expt["crystal"];
        json crystal_json =  elist_json_obj["crystal"][crystal_id];
        Crystal this_crystal(crystal_json);
        int scan_id = expt["scan"];
        int image_no = scans[scan_id]["image_range"][0];
        identifier_to_image_number[identifier] = image_no;
        identifier_to_crystal_model[identifier] = this_crystal;
    }

    auto id_col = reflections.column<int>("id");
    if (!id_col){
        logger.error("Column 'id' not found in reflection data.");
        return 1;
    }
    mdspan_type<int> ids = id_col.value();
    int num_reflections = 0;

    std::unordered_map<int, size_t> reflections_in_image;

    const std::vector<std::string>& identifiers = reflections.get_identifiers();
    const std::vector<uint64_t>& expt_ids = reflections.get_experiment_ids();

    // Ordered by keys i.e. image no
    std::unique_ptr<std::map<int, std::vector<Prediction>>> predictions_mapptr = std::make_unique<std::map<int, std::vector<Prediction>>>();
    std::mutex predictions_mutex;  // Mutex to protect the reflection intensities map
    std::unique_ptr<std::map<int, Simple6MosaicityParameterisation>> models_mapptr = std::make_unique<std::map<int, Simple6MosaicityParameterisation>>();
    std::mutex models_mutex;  // Mutex to protect the reflection intensities map
    
    auto max_it = std::max_element(
        ids.data_handle(),
        ids.data_handle() + ids.size()
    );

    int max_value = *max_it;
    std::exception_ptr eptr = nullptr;
    std::mutex eptr_mtx;
    {
        ThreadPool pool(num_cpu_threads);

        for (int index = 0; index < max_value; ++index) {
            pool.enqueue([&, index] {
            try {
                // get the data from the table

                std::vector<size_t> sel;
                for (int j=0;j<reflections.size();j++){
                    if (ids(j,0) == index){
                        sel.push_back(j);
                    }
                }
                ReflectionTable table = reflections.select(sel);
                std::string identifier = identifiers[index];
                

                std::vector<std::string> identifiers = table.get_identifiers();
                int image_no = identifier_to_image_number[identifier];
                reflections_in_image[image_no] = table.size();
                num_reflections += table.size();
                Crystal this_crystal = identifier_to_crystal_model[identifier];
                // make mdspans of data
                auto xyzobs_px_ = table.column<double>("xyzobs.px.value");
                if (!xyzobs_px_.has_value()) {
                    throw std::runtime_error(
                    "No xyzobs.px.value column found in input reflection table.");
                }
                mdspan_type<const double> xyzobs_px = xyzobs_px_.value();
                
                auto xyzcal_px_ = table.column<double>("xyzcal.px");
                if (!xyzcal_px_.has_value()) {
                    throw std::runtime_error(
                    "No xyzcal.px column found in input reflection table.");
                }
                mdspan_type<const double> xyzcal_px = xyzcal_px_.value();

                auto covariances_opt = table.column<double>("spot_covariance_2d_packed");
                if (!covariances_opt) {
                    logger.error("Column 'spot_covariance_2d_packed' not found in reflection data.");
                    return;
                }
                mdspan_type<const double> covariances = covariances_opt.value();

                auto intensities_opt = table.column<int>("intensity.sum.value");
                if (!intensities_opt) {
                    logger.error("Column 'intensity.sum.value' not found in reflection data.");
                    return;
                }
                mdspan_type<const int> strong_intensities = intensities_opt.value();

                auto miller_indices_opt = table.column<int>("miller_index");
                if (!miller_indices_opt) {
                    logger.error("Column 'miller_index' not found in reflection data.");
                    return;
                }
                mdspan_type<const int> miller_indices = miller_indices_opt.value();

                auto mobs_opt = table.column<double>("spot_mobs_2d");
                if (!mobs_opt) {
                    logger.error("Column 'spot_mobs_2d' not found in reflection data.");
                    return;
                }
                mdspan_type<const double> mobs = mobs_opt.value();

                // refine predict
                Simple6MosaicityParameterisation model = refine_mosaicity(xyzcal_px,
                                                                xyzobs_px,
                                                                covariances,
                                                                strong_intensities,
                                                                miller_indices,
                                                                mobs,
                                                                s0,
                                                                panel,
                                                                this_crystal.get_A_matrix());
                std::vector<Prediction> predictions = predict_ssx(model, s0, panel, this_crystal.get_A_matrix());
                logger.info("Predicted {} reflections on image {}", predictions.size(), image_no);
                if (predictions.size() > 0){
                    std::lock_guard<std::mutex> lock(predictions_mutex);
                    (*predictions_mapptr)[image_no] = std::move(predictions);
                    std::lock_guard<std::mutex> lock2(models_mutex);
                    (*models_mapptr)[image_no] = std::move(model);
                }
                
                } catch (...) {
                    std::lock_guard<std::mutex> lk(eptr_mtx);
                    if (!eptr) eptr = std::current_exception();
                }
            });
        }
    }
    if (eptr) std::rethrow_exception(eptr);


    std::size_t total_predictions = 0;
    for (const auto& [key, predictions] : *predictions_mapptr) {
        total_predictions += predictions.size();
    }
    logger.info("{} reflections predicted across {} unique images",
                total_predictions, reflections_in_image.size());
    num_reflections = total_predictions;

    const uint64_t background_pool_capacity =
    static_cast<uint64_t>(num_reflections) * background_pool_entries_per_reflection;
    if (background_pool_capacity > static_cast<uint64_t>(UINT32_MAX)) {
        logger.error(
          "Background pool of {} entries exceeds the 32-bit index limit of {} "
          "(--background.pool_entries_per_reflection times the reflection count)",
          background_pool_capacity,
          UINT32_MAX);
        return 1;
    }


    std::vector<BoundingBoxExtents> bboxes;
    std::vector<int> predicted_ids;
    std::vector<int> predicted_miller_index;
    std::vector<double> predicted_partialities;
    std::unordered_map<int, std::vector<size_t>> reflections_by_image;
    std::unordered_map<int, uint32_t> corner_tile_by_image;
    bboxes.reserve(num_reflections);
    predicted_ids.reserve(num_reflections);
    predicted_partialities.reserve(num_reflections);
    predicted_miller_index.reserve(num_reflections*3);
    uint64_t cumulative_index = 0;
    int current_id = 0;

    Eigen::Matrix<scalar_t, 3, 3, Eigen::RowMajor> d_matrix_flat =
        panel.get_d_matrix().cast<scalar_t>();
    std::vector<fastvec::Vector3D> s1_vectors_vec;
    std::vector<fastvec::Vector3D> s2_vectors_vec;
    std::vector<Vec2D> mubar_vec;
    std::vector<Mat2D> Sbar_inv_vec;
    s1_vectors_vec.reserve(num_reflections);
    s2_vectors_vec.reserve(num_reflections);
    fastvec::Vector3D s0_vec = to_vector3d(s0);
    scalar_t wavelength_scalar = static_cast<scalar_t>(wavelength);
    DetectorParameters det_params = make_detector_params(panel);

    // Map is ordered by keys
    double norm_s0 = 1.0/wavelength;
    for (const auto& [key, predictions] : *predictions_mapptr) {
        Simple6MosaicityParameterisation& model = (*models_mapptr)[key];
        Matrix3d sigma = model.sigma();
        
        for (const auto& p: predictions){
            bboxes.emplace_back(p.bbox_extent);
            const auto &bbox = p.bbox_extent;
            const uint32_t corner_grid =
                static_cast<uint32_t>(bbox.x_max - bbox.x_min + 1)
                * static_cast<uint32_t>(bbox.y_max - bbox.y_min + 1);
            uint32_t &image_max = corner_tile_by_image[key];
            image_max = std::max(image_max, corner_grid);
            predicted_ids.emplace_back(current_id);
            predicted_partialities.emplace_back(p.partiality);
            predicted_miller_index.emplace_back(p.h[0]);
            predicted_miller_index.emplace_back(p.h[1]);
            predicted_miller_index.emplace_back(p.h[2]);
            s1_vectors_vec.push_back(to_vector3d(p.s1));
            s2_vectors_vec.push_back(to_vector3d(p.s2));

            // Precalculate mubar and Sbar inv, as these are fixed for each reflection and we don't want
            // to have to recalculate in a hot loop on the gpu.
            const Eigen::Vector3d r = p.s2 - s0;
            const Eigen::Matrix3d R = compute_change_of_basis_operation(s0, p.s2);
            const Eigen::Matrix3d S = R * sigma * R.transpose();
            const Eigen::Vector3d mu = R * p.s2;
            double epsilon = norm_s0 - mu(2); 
            const Eigen::Vector2d mu1 = mu.head<2>();
            const Eigen::Matrix2d S11 = sigma.block<2, 2>(0, 0);
            const Eigen::Vector2d S12 = sigma.block<2, 1>(0, 2);
            const Eigen::RowVector2d S21 = sigma.block<1, 2>(2, 0);
            double S22 = sigma(2,2);
            const Eigen::Vector2d mubar = mu1 + S12 * (epsilon / S22);
            const Eigen::Matrix2d Sbar = S11 - (S12 / S22) * S21;
            const Eigen::Matrix2d Sbar_inv = Sbar.inverse();
            mubar_vec.push_back({mubar(0), mubar(1)});
            Mat2D Sbar_inv_mat;
            Sbar_inv_mat.xx = Sbar_inv(0, 0);
            Sbar_inv_mat.xy = Sbar_inv(0, 1);
            Sbar_inv_mat.yx = Sbar_inv(1, 0);
            Sbar_inv_mat.yy = Sbar_inv(1, 1);

            Sbar_inv_vec.push_back(Sbar_inv_mat);
        }
        
        std::vector<uint64_t> refl_indices;
        refl_indices.reserve(predictions.size());
        for (uint64_t i=cumulative_index;i<cumulative_index+predictions.size();++i){
            refl_indices.emplace_back(i);
        }
        cumulative_index += predictions.size();
        current_id += 1;
        reflections_by_image[key] = std::move(refl_indices);
    }

    /*for (size_t refl_id = 0; refl_id < num_reflections; ++refl_id) {
        const auto &bbox = bboxes[refl_id];

        // The bbox is half-open in x and y, so w×h is exactly the pixel
        // count and the corner grid is one larger on each axis.
        const uint32_t corner_grid =
          static_cast<uint32_t>(bbox.x_max - bbox.x_min + 1)
          * static_cast<uint32_t>(bbox.y_max - bbox.y_min + 1);

        // Add this reflection to all images it spans. The shoebox z
        // range is half-open [z_min, z_max), so z_max is excluded.
        for (int z = bbox.z_min; z < bbox.z_max; ++z) {
            reflections_by_image[z].push_back(refl_id);
            uint32_t &image_max = corner_tile_by_image[z];
            image_max = std::max(image_max, corner_grid);
        }
    }*/

    //
    //file is nexus file extracted from elist
    std::unique_ptr<Reader> reader_ptr;
    std::string file = "/scratch/whi10850/ssx_mx36015-64-axcp_idris_2492/idris_2492.nxs"; //FIXME
    reader_ptr = std::make_unique<H5Read>(file);
    Reader &reader = *reader_ptr;
    auto reader_mutex = std::mutex{};
    size_t bytes_per_pixel = reader.get_element_size();
    // Ensure this matches what we expect
    if (bytes_per_pixel != sizeof(pixel_t)) {
        fmt::print(
            "Error: Data type mismatch; This executable only accepts {} bit != {}\n",
            sizeof(pixel_t) * 8,
            bytes_per_pixel * 8);
        std::exit(bytes_per_pixel * 8);
    }
    uint32_t num_images = 3000;//reader.get_number_of_images();
    uint32_t height = reader.image_shape()[0];
    uint32_t width = reader.image_shape()[1];
    auto trusted_px_max = reader.get_trusted_range()[1];

    // Work out how many blocks this is
    dim3 gpu_thread_block_size{32, 16};
    dim3 blocks_dims{
        static_cast<unsigned int>(ceilf((float)width / gpu_thread_block_size.x)),
        static_cast<unsigned int>(ceilf((float)height / gpu_thread_block_size.y))};
    const int num_threads_per_block = gpu_thread_block_size.x * gpu_thread_block_size.y;
    const int num_blocks = blocks_dims.x * blocks_dims.y * blocks_dims.z;
    fmt::print("Image:       {:4d} x {:4d} = {} px\n", width, height, width * height);
    fmt::print("GPU Threads: {:4d} x {:<4d} = {}\n",
                gpu_thread_block_size.x,
                gpu_thread_block_size.y,
                num_threads_per_block);
    fmt::print("Blocks:      {:4d} x {:<4d} x {:2d} = {}\n",
                blocks_dims.x,
                blocks_dims.y,
                blocks_dims.z,
                num_blocks);
    fmt::print("Running with {} CPU threads\n", num_cpu_threads);


    auto all_images_start_time = std::chrono::high_resolution_clock::now();

    auto next_image = std::atomic<int>(0);
    auto completed_images = std::atomic<int>(0);

    auto cpu_sync = std::barrier{num_cpu_threads};
    double time_waiting_for_images = 0.0;

    #pragma region Prep GPU Data Buffers
    // What info do we have that we need:
    // bbox
    // assumed panel_id = 0 i.e. single panel
    // computing mask - need s1, s2 from refl table
    //                - need s0, covariance matrix, panel

    
    // Allocate and copy GPU buffers (shared across all threads)
    DeviceBuffer<scalar_t> d_d_matrix(9);
    DeviceBuffer<fastvec::Vector3D> d_s1_vectors(num_reflections);
    DeviceBuffer<fastvec::Vector3D> d_s2_vectors(num_reflections);
    DeviceBuffer<BoundingBoxExtents> d_bboxes(num_reflections);
    DeviceBuffer<Vec2D> d_mubar(num_reflections);
    DeviceBuffer<Mat2D> d_sbar_inv(num_reflections);
    DeviceBuffer<uint8_t> d_success(num_reflections);

    d_d_matrix.assign(d_matrix_flat.data());
    d_s1_vectors.assign(s1_vectors_vec.data());
    d_s2_vectors.assign(s2_vectors_vec.data());
    d_mubar.assign(mubar_vec.data());
    d_sbar_inv.assign(Sbar_inv_vec.data());
    // Test case, let's just loop over bboxes summing all pixels.
    d_bboxes.assign(bboxes.data());
    std::vector<uint8_t> success_init(num_reflections, 1);
    d_success.assign(success_init.data());

    // Two copies of the pool must fit beside the image buffers, since the
    // reduction sorts into a second one.
    {
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        constexpr size_t kOtherAllocationsMargin = size_t{256} << 20;
        const size_t entry_bytes = sizeof(unsigned long long) + sizeof(uint32_t);
        const size_t pool_bytes = 2 * background_pool_capacity * entry_bytes;
        const size_t image_bytes =
          static_cast<size_t>(num_cpu_threads) * width * height * sizeof(pixel_t);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t other_bytes = image_bytes + kOtherAllocationsMargin;
        if (pool_bytes + other_bytes > free_bytes) {
            const size_t room = free_bytes > other_bytes ? free_bytes - other_bytes : 0;
            const size_t fits =
              room / (2 * entry_bytes * std::max<size_t>(num_reflections, 1));
            logger.error(
              "The background pool needs {:.1f} GiB but {:.1f} GiB is free; set "
              "--background.pool_entries_per_reflection to {} or lower",
              pool_bytes / kGiB,
              room / kGiB,
              fits);
            return 1;
        }
    }
    DeviceBuffer<unsigned long long> d_background_pool_keys(background_pool_capacity);
    DeviceBuffer<uint32_t> d_background_pool_counts(background_pool_capacity);
    BackgroundPool background_pool;
    background_pool.keys = d_background_pool_keys.data();
    background_pool.counts = d_background_pool_counts.data();
    background_pool.capacity = static_cast<uint32_t>(background_pool_capacity);
    DeviceBuffer<uint32_t> d_background_spill(num_reflections);
    logger.info("Background pool: {} entries ({} per reflection), {:.1f} MiB",
                background_pool_capacity,
                background_pool_entries_per_reflection,
                background_pool_capacity
                  * (sizeof(unsigned long long) + sizeof(uint32_t))
                  / (1024.0 * 1024.0));
    // Reduced background estimates (written by compute_background()).
    DeviceBuffer<double> d_background_mean(num_reflections);
    DeviceBuffer<double> d_background_sum_value(num_reflections);
    DeviceBuffer<uint32_t> d_background_count(num_reflections);
    DeviceBuffer<uint8_t> d_background_success(num_reflections);

    

    // Upload the detector mask once (constant across all images).
    // non-zero = valid, 0 = masked. If the reader provides no mask,
    // treat every pixel as valid. Stored flat (width*height), indexed
    // `gy * width + gx`.


    // What do we need: s1, s2, s0, panel for pixel lab coord, sigma,
    // Could precompute pixel to lab coords?

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

    //DetectorParameters det_params = make_detector_params(panel);

    //fastvec::Vector3D s0_vec = to_vector3d(s0);

    DeviceBuffer<accumulator_t> d_foreground_sum(num_reflections);
    DeviceBuffer<uint32_t> d_foreground_count(num_reflections);

    cuda_check(
      cudaMemset(d_foreground_sum.data(), 0, num_reflections * sizeof(accumulator_t)));
    cuda_check(
      cudaMemset(d_foreground_count.data(), 0, num_reflections * sizeof(uint32_t)));
    // An all-ones byte fill makes every key BG_POOL_EMPTY_KEY.
    cuda_check(cudaMemset(d_background_pool_keys.data(),
                          0xFF,
                          background_pool_capacity * sizeof(unsigned long long)));
    cuda_check(cudaMemset(
      d_background_pool_counts.data(), 0, background_pool_capacity * sizeof(uint32_t)));
    cuda_check(
      cudaMemset(d_background_spill.data(), 0, num_reflections * sizeof(uint32_t)));

    // assume already calculated corrections (LP, qe)
    #pragma endregion Prep GPU Data Buffers

    #pragma region Thread launch
    // We are going to iterate through the images,
    // doing refine, predict and then integrate on gpu
    // The result we will store is the reflection table (and mosaicity models?).
    // For testing, assume we have already refined and predicted.

    // Spawn the reader threads
    std::vector<std::jthread> threads;
    for (int thread_id = 0; thread_id < num_cpu_threads; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            auto stop_token = global_stop.get_token();
            CudaStream stream;

            auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height);
            //auto host_results = make_cuda_pinned_malloc<uint8_t>(width * height);
            auto device_image = PitchedMalloc<pixel_t>(width, height);
            /*auto device_results =
                PitchedMalloc<uint8_t>(make_cuda_malloc<uint8_t[]>(mask.pitch * height),
                                        width,
                                        height,
                                        mask.pitch);*/

            // Buffer for reading compressed chunk data in
            auto raw_chunk_buffer =
                std::vector<uint8_t>(width * height * sizeof(pixel_t));

            // Let all threads do setup tasks before reading starts
            cpu_sync.arrive_and_wait();
            //CudaEvent start, copy, post, postcopy, end;

            // Get the time the lastimage was received to avoid waiting for too long
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
                    // Do we really need the wait check in integration?
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
                            logger.warn("Timeout waiting for image {}",
                                        image_num);
                            global_stop.request_stop();
                            break;
                        }

                        // Sleep for a bit to avoid busy-waiting
                        //std::this_thread::sleep_for(100ms);
                    }

                    if (stop_token.stop_requested()) {
                        break;
                    }

                    // If the image is available, update the last image received time
                    last_image_received = std::chrono::high_resolution_clock::now();

                    time_waiting_for_images +=
                        std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now()
                        - swmr_wait_start_time)
                        .count();
                }
                // Sized buffer for the actual data read from file
                std::span<uint8_t> buffer;
                // Fetch the image data from the reader.. loop in case we need to wait
                while (true) {
                    {
                        std::scoped_lock lock(reader_mutex);
                        buffer =
                            reader.get_raw_chunk(image_num, raw_chunk_buffer);
                    }
                    // /dev/shm we might not have an atomic write
                    if (buffer.size() == 0) {
                        logger.warn(
                            "\033[1mRace Condition?!?? Got buffer size 0 for image "
                            "{}. Sleeping.\033[0m",
                            image_num);
                        //std::this_thread::sleep_for(100ms);
                        continue;
                    }
                    break;
                }

    #pragma region Decompression
                // Decompress this data, outside of the mutex.
                // We do this here rather than in the reader, because we
                // anticipate that we will want to eventually offload
                // the decompression
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

                //start.record(stream);
                // Copy the image to GPU
                cuda_check(cudaMemcpy2DAsync(device_image.get(),
                                                device_image.pitch_bytes(),
                                                host_image.get(),
                                                width * sizeof(pixel_t),
                                                width * sizeof(pixel_t),
                                                height,
                                                cudaMemcpyHostToDevice,
                                                stream));
                //copy.record(stream);
    #pragma endregion Decompression
                // Get reflection indices for this image and copy to device
                const auto &refl_indices = reflections_by_image[image_num];
                size_t num_refls_this_image = refl_indices.size();

                // Allocate per-image device buffer for reflection indices
                DeviceBuffer<size_t> d_reflection_indices(num_refls_this_image);
                d_reflection_indices.assign(refl_indices.data());
                logger.info("Launching GPU kernel for image {} with {} reflections",
                            image_num,
                            num_refls_this_image);
                // For test, just sum all pixels in bbox.


                // If precompute Sbar_inv and mubar per refl on cpu....
                // Then much easier.

                compute_ellipsoid_integrate(device_image.get(),
                                    device_image.pitch_bytes(),
                                    width,
                                    height,
                                    image_num,
                                    d_d_matrix.data(),
                                    wavelength_scalar,
                                    det_params,
                                    s0_vec,
                                    d_bboxes.data(),
                                    d_reflection_indices.data(),
                                    d_s2_vectors.data(),
                                    d_mubar.data(),
                                    d_sbar_inv.data(),
                                    num_refls_this_image,
                                    d_mask.data(),
                                    corner_tile_by_image.at(image_num),
                                    d_foreground_sum.data(),
                                    d_foreground_count.data(),
                                    background_pool,
                                    d_background_spill.data(),
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
    int n_integrated_images = reflections_by_image.size();
    logger.info("{} images processed in {:.2f} s ({:.1f} fps)",
                int(n_integrated_images),
                total_time,
                n_integrated_images / total_time);
    if (time_waiting_for_images < 10) {
        logger.info("Total time waiting for images: {:.0f} ms",
                    time_waiting_for_images * 1000);
    } else {
        logger.info("Total time waiting for images: {:.2f} s", time_waiting_for_images);
    }
    cuda_check(cudaDeviceSynchronize());

    logger.info("Reducing background histograms for {} reflections", num_reflections);
    const uint32_t background_pool_occupied =
      compute_background(background_model,
                         background_pool,
                         d_background_spill.data(),
                         num_reflections,
                         d_background_mean.data(),
                         d_background_sum_value.data(),
                         d_background_count.data(),
                         d_background_success.data(),
                         0);
    // How full the pool got is the margin before it would start to spill.
    logger.info("Background pool: {} of {} entries used ({:.1f}%)",
                background_pool_occupied,
                background_pool_capacity,
                100.0 * background_pool_occupied / background_pool_capacity);
    cuda_check(cudaDeviceSynchronize());

    logger.info("Finalizing integration for {} reflections", num_reflections);
    // Allocate host buffers for results
    std::vector<accumulator_t> h_foreground_sum(num_reflections);
    std::vector<uint32_t> h_foreground_count(num_reflections);
    std::vector<double> h_background_mean(num_reflections);
    std::vector<double> h_background_sum_value(num_reflections);
    std::vector<uint32_t> h_background_count(num_reflections);
    std::vector<uint32_t> h_background_spill(num_reflections);
    std::vector<uint8_t> h_background_success(num_reflections);
    std::vector<uint8_t> h_success(num_reflections);
    // Copy results from device to host
    d_foreground_sum.extract(h_foreground_sum.data());
    d_foreground_count.extract(h_foreground_count.data());
    d_background_mean.extract(h_background_mean.data());
    d_background_sum_value.extract(h_background_sum_value.data());
    d_background_count.extract(h_background_count.data());
    d_background_spill.extract(h_background_spill.data());
    d_background_success.extract(h_background_success.data());
    d_success.extract(h_success.data());

    {
        size_t spilling = 0;
        for (size_t i = 0; i < num_reflections; ++i) {
            if (h_background_spill[i] > 0) {
                spilling++;
            }
        }
        if (spilling > 0) {
            logger.warn(
              "{} of {} reflection(s) lost background pixels to a full pool; set "
              "--background.pool_entries_per_reflection above {}",
              spilling,
              num_reflections,
              background_pool_entries_per_reflection);
        }
    }

    // Compute final intensities for each reflection
    std::vector<double> intensities(num_reflections);
    std::vector<double> variances(num_reflections);
    std::vector<double> sigmas(num_reflections);       // σ(I) = √Var(I)
    std::vector<double> backgrounds(num_reflections);  // b̄  (background mean per pixel)
    std::vector<double> background_sigmas(num_reflections);  // σ(b̄)
    std::vector<uint8_t> success_final(num_reflections, 0);

    size_t valid_reflections = 0;
    size_t background_failures = 0;  // fg present but background estimate rejected
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

        // Tally rejected estimates (no inlier pixels, too few pixels for the
        // GLM, or pixels lost to a full pool) for the warning below.
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

        const bool ok = h_success[i] && h_background_success[i];
        success_final[i] = ok ? 1 : 0;
        if (ok) {
            valid_reflections++;
        }
    }

    logger.info("Summation integration complete: {} valid reflections out of {}",
                valid_reflections,
                num_reflections);

    if (background_failures > 0) {
        logger.warn(
          "Background estimate rejected for {} of {} reflections with "
          "foreground pixels; any that lost pixels to a full pool are counted above",
          background_failures,
          num_reflections);
    }
    logger.info("Building output reflection table");
    // Build output reflection table.
    std::vector<int> nfg_out(num_reflections);
    std::vector<int> nbg_out(num_reflections);
    std::vector<double> bg_sum_out(num_reflections);
    std::vector<double> bg_mean_out(num_reflections);
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

    }

    std::vector<std::string> final_identifiers = reflections.get_identifiers();
    std::vector<uint64_t> ids_in = reflections.get_experiment_ids();
    ReflectionTable integrated_data(ids_in, final_identifiers);
    integrated_data.add_column("intensity.sum.value", num_reflections, 1, intensities);
    integrated_data.add_column(
      "intensity.sum.variance", num_reflections, 1, variances);
    integrated_data.add_column("id", num_reflections, 1, predicted_ids);
    integrated_data.add_column("partiality", num_reflections, 1, predicted_partialities);
    integrated_data.add_column("miller_index", num_reflections, 3, predicted_miller_index);
    integrated_data.add_column("num_pixels.background", num_reflections, 1, nbg_out);
    integrated_data.add_column("num_pixels.foreground", num_reflections, 1, nfg_out);
    integrated_data.add_column("background.sum.value", num_reflections, 1, bg_sum_out);
    integrated_data.add_column("background.mean", num_reflections, 1, bg_mean_out);
    constexpr size_t IntegratedSum = (1 << 8);
    std::vector<std::size_t> final_flags(num_reflections, IntegratedSum);
    integrated_data.add_column("flags", num_reflections, 1, final_flags);

    std::vector<bool> success_bool(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i)
        success_bool[i] = success_final[i] != 0;
    ReflectionTable success_data = integrated_data.select(success_bool);
    int n_integrated =
      success_data.column<double>("intensity.sum.value").value().extent(0);
    logger.info("Writing {} integrated reflections to {}", n_integrated, "integrated.refl");
    success_data.write("integrated.refl");

    // Add profile models to an integrated.expt list.

    return 0;
}