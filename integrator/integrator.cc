/**
  * @file integrator.cc
 */

#include "integrator.cuh"

#include <bitshuffle.h>

#include <Eigen/Dense>
#include <atomic>
#include <barrier>
#include <chrono>
#include <csignal>
#include <dx2/beam.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>
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
#include "extent.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"
#include "kabsch.cuh"
#include "version.hpp"

using namespace std::chrono_literals;

// Define a 2D mdspan type alias for convenience
using mdspan_2d =
  std::experimental::mdspan<scalar_t, std::experimental::dextents<size_t, 2>>;

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
 * @brief Wait for a file/path to be ready for reading
 */
void wait_for_ready_for_read(const std::string &path,
                             std::function<bool(const std::string &)> checker,
                             float timeout = 120.0f) {
    if (!checker(path)) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto message_prefix =
          fmt::format("Waiting for \033[1;35m{}\033[0m to be ready for read", path);
        std::vector<std::string> ball = {
          "( ●    )",
          "(  ●   )",
          "(   ●  )",
          "(    ● )",
          "(     ●)",
          "(    ● )",
          "(   ●  )",
          "(  ●   )",
          "( ●    )",
          "(●     )",
        };
        int i = 0;
        while (!checker(path)) {
            auto wait_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                               std::chrono::high_resolution_clock::now() - start_time)
                               .count();
            fmt::print("\r{}  {} [{:4.1f} s] ", message_prefix, ball[i], wait_time);
            i = (i + 1) % ball.size();
            std::cout << std::flush;

            if (wait_time > timeout) {
                fmt::print("\nError: Waited too long for read availability\n");
                std::exit(1);
            }
            std::this_thread::sleep_for(80ms);
        }
        fmt::print("\n");
    }
}

/**
 * @brief Structure to hold detector parameters for GPU kernels
 * 
 * This struct packages all the detector-specific parameters needed for
 * correct coordinate transformations on the GPU, including parallax correction.
 */
struct DetectorParameters {
    scalar_t pixel_size[2];       // [x_size, y_size] in mm
    bool parallax_correction;     // Whether to apply parallax correction
    scalar_t mu;                  // Absorption coefficient
    scalar_t thickness;           // Detector thickness in mm
    fastvec::Vector3D fast_axis;  // Detector fast axis direction
    fastvec::Vector3D slow_axis;  // Detector slow axis direction
    fastvec::Vector3D origin;     // Detector origin position
};

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
          .help("Number of parallel reader threads")
          .default_value<uint32_t>(1)
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
    float sigma_m = parser.get<float>("sigma_m");
    float sigma_b = parser.get<float>("sigma_b");

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

    // Display column names
    std::string column_names_str;
    for (const auto &name : reflections.get_column_names()) {
        column_names_str += "\n\t- " + name;
    }
    logger.info("Column names: {}", column_names_str);

#pragma region Data preparation

    // Extract required columns and dereference optionals
    auto s1_vectors_opt = reflections.column<double>("s1");
    if (!s1_vectors_opt) {
        logger.error("Column 's1' not found in reflection data.");
        return 1;
    }
    auto s1_vectors = *s1_vectors_opt;

    auto phi_column_opt = reflections.column<double>("xyzcal.mm");
    if (!phi_column_opt) {
        logger.error("Column 'xyzcal.mm' not found for phi positions.");
        return 1;
    }
    auto phi_column = *phi_column_opt;

    // TODO: Improve this hacky conversion from double to scalar_t (float)
    std::vector<scalar_t> s1_vectors_converted_data(s1_vectors.extent(0) * 3);
    mdspan_type<scalar_t> s1_vectors_converted(
      s1_vectors_converted_data.data(), s1_vectors.extent(0), 3);
    size_t num_reflections = s1_vectors.extent(0);

    // Direct pointer access conversion loop -> compiler should optimize this
    const double *src = s1_vectors.data_handle();
    scalar_t *dst = s1_vectors_converted_data.data();
    for (size_t i = 0; i < num_reflections * 3; ++i) {
        dst[i] = static_cast<scalar_t>(src[i]);
    }

    std::vector<scalar_t> phi_positions_converted_data(phi_column.extent(0));
    // mdspan_type<scalar_t> phi_positions_converted(phi_positions_converted_data.data(), phi_column.extent(0));
    // Direct pointer access conversion loop -> compiler should optimize this
    src = phi_column.data_handle();
    dst = phi_positions_converted_data.data();
    for (size_t i = 0; i < phi_column.extent(0); ++i) {
        dst[i] = static_cast<scalar_t>(src[i]);
    }

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
    Experiment<MonochromaticBeam> expt;
    try {
        expt = Experiment<MonochromaticBeam>(elist_json_obj);
    } catch (const std::invalid_argument &ex) {
        logger.error(
          "Failed to construct Experiment from '{}': {}", experiment_file, ex.what());
        return 1;
    }

    // Extract experimental components
    MonochromaticBeam beam = expt.beam();
    Goniometer gonio = expt.goniometer();
    const Panel &panel = expt.detector().panels()[0];  // Assuming single panel detector
    const Scan &scan = expt.scan();

    // Extract vectors and parameters
    Eigen::Vector3d s0 = beam.get_s0();
    Eigen::Vector3d rotation_axis = gonio.get_rotation_axis();
    // size_t num_reflections = s1_vectors.extent(0);
    const auto [osc_start, osc_width] = scan.get_oscillation();
    int image_range_start = scan.get_image_range()[0];
    double wl = beam.get_wavelength();

#pragma endregion Data preparation

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

    // Map reflections by z layer (image number)
    logger.info("Mapping reflections by image number (z layer)");
    std::unordered_map<int, std::vector<size_t>> reflections_by_image;

    for (size_t refl_id = 0; refl_id < num_reflections; ++refl_id) {
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

    // Get threading parameters
    uint32_t num_cpu_threads = parser.get<uint32_t>("threads");
    if (num_cpu_threads < 1) {
        logger.error("Thread count must be >= 1");
        return 1;
    }
    logger.info("Running with {} CPU threads", num_cpu_threads);

    // Set up image reader
    const auto images_file = parser.images();
    std::unique_ptr<Reader> reader_ptr;

    // Wait for read-readiness
    if (!std::filesystem::exists(images_file)) {
        wait_for_ready_for_read(
          images_file,
          [](const std::string &s) { return std::filesystem::exists(s); },
          wait_timeout);
    }

    if (std::filesystem::is_directory(images_file)) {
        wait_for_ready_for_read(images_file, is_ready_for_read<SHMRead>, wait_timeout);
        reader_ptr = std::make_unique<SHMRead>(images_file);
    } else if (images_file.ends_with(".cbf")) {
        logger.error("CBF reading not yet supported in integrator mode");
        return 1;
    } else {
        wait_for_ready_for_read(images_file, is_ready_for_read<H5Read>, wait_timeout);
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

    // Spawn the reader threads
    std::vector<std::jthread> threads;
    for (int thread_id = 0; thread_id < num_cpu_threads; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            auto stop_token = global_stop.get_token();

            // Full image buffers for decompression
            auto decompressed_image = make_cuda_pinned_malloc<pixel_t>(width * height);
            auto raw_chunk_buffer =
              std::vector<uint8_t>(width * height * sizeof(pixel_t));

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

                // Decompress the data
                switch (reader.get_raw_chunk_compression()) {
                case Reader::ChunkCompression::BITSHUFFLE_LZ4:
                    bshuf_decompress_lz4(buffer.data() + 12,
                                         decompressed_image.get(),
                                         width * height,
                                         sizeof(pixel_t),
                                         0);
                    break;
                case Reader::ChunkCompression::BYTE_OFFSET_32:
                    decompress_byte_offset<pixel_t>(
                      buffer,
                      {decompressed_image.get(),
                       static_cast<std::span<pixel_t>::size_type>(width * height)});
                    break;
                }

                // TODO: image processing here
                // decompressed_image.get() contains the decompressed pixel data (width * height pixels)
                // reflections_by_image[image_num] contains the reflection IDs for this image

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

#pragma endregion Image Reading and Threading

    return 0;
}
#pragma endregion Application Entry

#pragma region Bbox computation
/*
    // Compute new bounding boxes using Kabsch coordinate system

    logger.info("Computing new Kabsch bounding boxes for {} reflections",
                num_reflections);

    // Create output array for bounding boxes (6 values per reflection)
    std::vector<scalar_t> computed_bbox_data(num_reflections * 6);

    // TODO: Make these conversions better

    // Convert s1_vectors from double to scalar_t (float) format
    std::vector<fastvec::Vector3D> s1_vectors_converted(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        s1_vectors_converted[i] =
          fastvec::make_vector3d(static_cast<scalar_t>(s1_vectors(i, 0)),
                                 static_cast<scalar_t>(s1_vectors(i, 1)),
                                 static_cast<scalar_t>(s1_vectors(i, 2)));
    }

    // Convert phi values from double to scalar_t
    std::vector<scalar_t> phi_values_converted(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        phi_values_converted[i] =
          static_cast<scalar_t>(phi_column(i, 2));  // Extract phi (z-component)
    }

    // Extract detector parameters (assuming you add accessors to Panel class)
    DetectorParameters detector_params;
    auto pixel_size_array = panel.get_pixel_size();
    detector_params.pixel_size[0] = static_cast<scalar_t>(pixel_size_array[0]);
    detector_params.pixel_size[1] = static_cast<scalar_t>(pixel_size_array[1]);

    // TODO: Add these accessors to Panel class
    detector_params.parallax_correction = panel.has_parallax_correction();
    detector_params.mu = static_cast<scalar_t>(panel.get_mu());
    detector_params.thickness = static_cast<scalar_t>(panel.get_thickness());

    // Convert geometry vectors to CUDA format
    auto fast_axis_eigen = panel.get_fast_axis();
    auto slow_axis_eigen = panel.get_slow_axis();
    auto origin_eigen = panel.get_origin();

    detector_params.fast_axis =
      fastvec::make_vector3d(static_cast<scalar_t>(fast_axis_eigen.x()),
                             static_cast<scalar_t>(fast_axis_eigen.y()),
                             static_cast<scalar_t>(fast_axis_eigen.z()));
    detector_params.slow_axis =
      fastvec::make_vector3d(static_cast<scalar_t>(slow_axis_eigen.x()),
                             static_cast<scalar_t>(slow_axis_eigen.y()),
                             static_cast<scalar_t>(slow_axis_eigen.z()));
    detector_params.origin =
      fastvec::make_vector3d(static_cast<scalar_t>(origin_eigen.x()),
                             static_cast<scalar_t>(origin_eigen.y()),
                             static_cast<scalar_t>(origin_eigen.z()));

    // Get D-matrix inverse
    auto d_matrix_inv = panel.get_d_matrix().inverse();
    std::vector<scalar_t> d_matrix_inv_flat(9);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            d_matrix_inv_flat[i * 3 + j] = static_cast<scalar_t>(d_matrix_inv(i, j));
        }
    }

    // Compute new bounding boxes using CUDA with proper detector parameters
    compute_bbox_extent(
      s1_vectors_converted.data(),
      phi_values_converted.data(),
      fastvec::make_vector3d(s0.x(), s0.y(), s0.z()),
      fastvec::make_vector3d(rotation_axis.x(), rotation_axis.y(), rotation_axis.z()),
      sigma_b,
      sigma_m,
      static_cast<scalar_t>(osc_start),
      static_cast<scalar_t>(osc_width),
      image_range_start,
      scan.get_image_range()[1],  // image_range_end
      static_cast<scalar_t>(beam.get_wavelength()),
      d_matrix_inv_flat.data(),
      detector_params.pixel_size,
      detector_params.parallax_correction,
      detector_params.mu,
      detector_params.thickness,
      detector_params.fast_axis,
      detector_params.slow_axis,
      detector_params.origin,
      computed_bbox_data.data(),
      num_reflections);

    // Convert to reflection table format for comparison
    // Data is already in the correct flat format from compute_bbox_extent
    logger.info("Bounding box computation completed");

    // Compare with existing bounding boxes
    logger.info("Comparing computed bounding boxes with existing bbox column");
    logger.trace("First 5 bounding box comparisons:");
    for (size_t i = 0; i < std::min<size_t>(5, num_reflections); ++i) {
        // Existing bbox
        double ex_x_min = bbox_column(i, 0);
        double ex_x_max = bbox_column(i, 1);
        double ex_y_min = bbox_column(i, 2);
        double ex_y_max = bbox_column(i, 3);
        int ex_z_min = static_cast<int>(bbox_column(i, 4));
        int ex_z_max = static_cast<int>(bbox_column(i, 5));

        // Computed bbox (from flat array)
        double comp_x_min = computed_bbox_data[i * 6 + 0];
        double comp_x_max = computed_bbox_data[i * 6 + 1];
        double comp_y_min = computed_bbox_data[i * 6 + 2];
        double comp_y_max = computed_bbox_data[i * 6 + 3];
        int comp_z_min = static_cast<int>(computed_bbox_data[i * 6 + 4]);
        int comp_z_max = static_cast<int>(computed_bbox_data[i * 6 + 5]);

        logger.trace("bbox[{}]: existing x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] z=[{},{}]",
                     i,
                     ex_x_min,
                     ex_x_max,
                     ex_y_min,
                     ex_y_max,
                     ex_z_min,
                     ex_z_max);
        logger.trace("bbox[{}]: computed x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] z=[{},{}]",
                     i,
                     comp_x_min,
                     comp_x_max,
                     comp_y_min,
                     comp_y_max,
                     comp_z_min,
                     comp_z_max);
    }

#pragma endregion Bbox computation
#pragma region Kabsch tf

    logger.info(
      "Computing Kabsch coordinates for voxel centers within existing bounding boxes");

    std::vector<double> voxel_kabsch_coords;  // ε₁, ε₂, ε₃ for each voxel center
    std::vector<int> voxel_reflection_ids;    // Which reflection each voxel belongs to
    std::vector<double> voxel_positions;      // x, y, z positions for each voxel center
    std::vector<double> voxel_s1_lengths;     // |s₁| for each voxel center

    // Convert global vectors to CUDA format once
    fastvec::Vector3D s0_cuda = fastvec::make_vector3d(s0.x(), s0.y(), s0.z());
    fastvec::Vector3D rotation_axis_cuda =
      fastvec::make_vector3d(rotation_axis.x(), rotation_axis.y(), rotation_axis.z());

    // Process each reflection's existing bounding box
    for (size_t refl_id = 0; refl_id < num_reflections; ++refl_id) {
        // Extract bounding box from existing column (format: x_min, x_max, y_min, y_max, z_min, z_max)
        double x_min = bbox_column(refl_id, 0);
        double x_max = bbox_column(refl_id, 1);
        double y_min = bbox_column(refl_id, 2);
        double y_max = bbox_column(refl_id, 3);
        int z_min = static_cast<int>(bbox_column(refl_id, 4));
        int z_max = static_cast<int>(bbox_column(refl_id, 5));

        // Debug: Calculate expected voxel count
        int x_count = static_cast<int>(x_max) - static_cast<int>(x_min);
        int y_count = static_cast<int>(y_max) - static_cast<int>(y_min);
        int z_count = z_max - z_min;
        int expected_voxels = x_count * y_count * z_count;

        logger.trace(
          "Reflection {}: bbox x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] z=[{},{}]",
          refl_id,
          x_min,
          x_max,
          y_min,
          y_max,
          z_min,
          z_max);
        logger.trace("Expected voxels: {} x {} x {} = {}",
                     x_count,
                     y_count,
                     z_count,
                     expected_voxels);

        // Get reflection centroid data
        Eigen::Vector3d s1_c_eigen(
          s1_vectors(refl_id, 0), s1_vectors(refl_id, 1), s1_vectors(refl_id, 2));
        fastvec::Vector3D s1_c_cuda =
          fastvec::make_vector3d(s1_c_eigen.x(), s1_c_eigen.y(), s1_c_eigen.z());
        double phi_c = phi_column(refl_id, 2);

        logger.trace(
          "Processing reflection {} with existing bbox x=[{:.1f},{:.1f}] "
          "y=[{:.1f},{:.1f}] z=[{},{}]",
          refl_id,
          x_min,
          x_max,
          y_min,
          y_max,
          z_min,
          z_max);

        // Collect all voxel data for this reflection for batch processing
        std::vector<fastvec::Vector3D> batch_s_pixels;
        std::vector<scalar_t> batch_phi_pixels;
        std::vector<std::tuple<double, double, double>> batch_voxel_coords;

        // Iterate through all voxel centers in the bounding box
        for (int z = z_min; z < z_max; ++z) {
            double phi_pixel =
              osc_start + (z - image_range_start + 1.5) * osc_width / 180.0 * M_PI;

            for (int y = static_cast<int>(y_min); y < static_cast<int>(y_max); ++y) {
                for (int x = static_cast<int>(x_min); x < static_cast<int>(x_max);
                     ++x) {
                    // Use voxel center coordinates
                    // double voxel_x = x + 0.5;
                    // double voxel_y = y + 0.5;
                    double voxel_z = 1;  //z + 0.5;

                    std::array<double, 2> xy_mm = panel.px_to_mm(
                      static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5);

                    double voxel_x = xy_mm[0];
                    double voxel_y = xy_mm[1];

                    // Convert detector coordinates to lab coordinate using panel geometry
                    Vector3d lab_coord =
                      panel.get_d_matrix() * Vector3d(voxel_x, voxel_y, 1.0);

                    // logger.trace(
                    //   "Voxel center ({}, {}, {}): lab_coord = ({:.6f}, {:.6f}, {:.6f})",
                    //   voxel_x,
                    //   voxel_y,
                    //   voxel_z,
                    //   lab_coord.x(),
                    //   lab_coord.y(),
                    //   lab_coord.z());

                    // Convert lab coordinate to reciprocal space vector
                    Eigen::Vector3d s_pixel_eigen = lab_coord.normalized() / wl;
                    fastvec::Vector3D s_pixel_cuda = fastvec::make_vector3d(
                      s_pixel_eigen.x(), s_pixel_eigen.y(), s_pixel_eigen.z());

                    // Store for batch processing
                    batch_s_pixels.push_back(s_pixel_cuda);
                    batch_phi_pixels.push_back(static_cast<scalar_t>(phi_pixel));
                    batch_voxel_coords.emplace_back(voxel_x, voxel_y, voxel_z);
                }
            }
        }

        // Process all voxels for this reflection with CUDA if we have any
        if (!batch_s_pixels.empty()) {
            size_t num_voxels = batch_s_pixels.size();

            // Output arrays
            std::vector<fastvec::Vector3D> batch_eps_results(num_voxels);
            std::vector<scalar_t> batch_s1_len_results(num_voxels);

            // Call CUDA function for voxel processing
            compute_kabsch_transform(batch_s_pixels.data(),
                                     batch_phi_pixels.data(),
                                     s1_c_cuda,
                                     static_cast<scalar_t>(phi_c),
                                     s0_cuda,
                                     rotation_axis_cuda,
                                     batch_eps_results.data(),
                                     batch_s1_len_results.data(),
                                     num_voxels);

            // Store results
            for (size_t i = 0; i < num_voxels; ++i) {
                const auto &[voxel_x, voxel_y, voxel_z] = batch_voxel_coords[i];
                const auto &eps = batch_eps_results[i];

                voxel_kabsch_coords.insert(voxel_kabsch_coords.end(),
                                           {eps.x, eps.y, eps.z});
                voxel_reflection_ids.push_back(static_cast<int>(refl_id));
                voxel_positions.insert(voxel_positions.end(),
                                       {voxel_x, voxel_y, voxel_z});
                voxel_s1_lengths.push_back(
                  static_cast<double>(batch_s1_len_results[i]));
            }
        }
    }

    size_t actual_voxels = voxel_reflection_ids.size();
    logger.info("Processed {} voxel centers, computed {} Kabsch coordinates",
                actual_voxels,
                voxel_kabsch_coords.size() / 3);

    // Add debugging information
    logger.info(
      "Vector sizes: kabsch_coords={}, reflection_ids={}, positions={}, s1_lengths={}",
      voxel_kabsch_coords.size(),
      voxel_reflection_ids.size(),
      voxel_positions.size(),
      voxel_s1_lengths.size());

    logger.info(
      "Expected sizes: kabsch_coords={}, reflection_ids={}, positions={}, "
      "s1_lengths={}",
      actual_voxels * 3,
      actual_voxels,
      actual_voxels * 3,
      actual_voxels);

#pragma endregion Kabsch tf
#pragma region Application Output

    // Add computed bounding boxes to reflection table for comparison
    // computed_bbox_data is already std::vector<double>
    reflections.add_column("computed_bbox",
                           std::vector<size_t>{num_reflections, 6},
                           computed_bbox_data);

    // Create voxel data table
    ReflectionTable voxel_table;
    voxel_table.add_column(
      "kabsch_coordinates", std::vector<size_t>{actual_voxels, 3}, voxel_kabsch_coords);
    voxel_table.add_column(
      "reflection_id",
      std::vector<size_t>{actual_voxels, 1},
      std::vector<double>(voxel_reflection_ids.begin(), voxel_reflection_ids.end()));
    voxel_table.add_column(
      "pixel_coordinates", std::vector<size_t>{actual_voxels, 3}, voxel_positions);
    voxel_table.add_column(
      "voxel_s1_length", std::vector<size_t>{actual_voxels, 1}, voxel_s1_lengths);

    // Write output files
    reflections.write("output_reflections.h5");
    voxel_table.write("voxel_kabsch_data.h5");

    logger.info("Results saved to output_reflections.h5 and voxel_kabsch_data.h5");
*/
#pragma endregion Application Output