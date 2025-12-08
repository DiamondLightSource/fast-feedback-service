/**
  * @file integrator.cc
 */

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
#include "ffs_logger.hpp"
#include "h5read.h"
#include "math/device_precision.cuh"
#include "math/math_utils.cuh"
#include "math/vector3d.cuh"
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
    // std::vector<BoundingBoxExtents> computed_bboxes =
    //   compute_kabsch_bounding_boxes(s0,
    //                                 rotation_axis,
    //                                 s1_vectors,
    //                                 phi_column,
    //                                 num_reflections,
    //                                 sigma_b,
    //                                 sigma_m,
    //                                 panel,
    //                                 scan,
    //                                 beam);

    logger.info("Bounding box computation completed");

    // Convert BoundingBoxExtents to flat array format for storage
    // std::vector<double> computed_bbox_data(num_reflections * 6);
    // for (size_t i = 0; i < num_reflections; ++i) {
    //     const int step = 6 * i;
    //     computed_bbox_data[step + 0] = computed_bboxes[i].x_min;
    //     computed_bbox_data[step + 1] = computed_bboxes[i].x_max;
    //     computed_bbox_data[step + 2] = computed_bboxes[i].y_min;
    //     computed_bbox_data[step + 3] = computed_bboxes[i].y_max;
    //     computed_bbox_data[step + 4] = static_cast<double>(computed_bboxes[i].z_min);
    //     computed_bbox_data[step + 5] = static_cast<double>(computed_bboxes[i].z_max);
    // }

    // Map reflections by z layer (image number)
    // logger.info("Mapping reflections by image number (z layer)");
    // std::unordered_map<int, std::vector<size_t>> reflections_by_image;

    // for (size_t refl_id = 0; refl_id < num_reflections; ++refl_id) {
    //     const auto &bbox = computed_bboxes[refl_id];

    //     // Add this reflection to all images it spans
    //     for (int z = bbox.z_min; z <= bbox.z_max; ++z) {
    //         reflections_by_image[z].push_back(refl_id);
    //     }
    // }

    // logger.info("Reflections mapped across {} unique images",
    //             reflections_by_image.size());

    // Log some statistics about the mapping
    // if (!reflections_by_image.empty()) {
    //     size_t min_refls_per_image = std::numeric_limits<size_t>::max();
    //     size_t max_refls_per_image = 0;
    //     size_t total_refls = 0;

    //     for (const auto &[image, refls] : reflections_by_image) {
    //         min_refls_per_image = std::min(min_refls_per_image, refls.size());
    //         max_refls_per_image = std::max(max_refls_per_image, refls.size());
    //         total_refls += refls.size();
    //     }

    //     double avg_refls_per_image =
    //       static_cast<double>(total_refls) / reflections_by_image.size();
    //     logger.info("Reflections per image: min={}, max={}, avg={:.1f}",
    //                 min_refls_per_image,
    //                 max_refls_per_image,
    //                 avg_refls_per_image);
    // }

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
    // if (!std::filesystem::exists(images_file)) {
    //     wait_for_ready_for_read(
    //       images_file,
    //       [](const std::string &s) { return std::filesystem::exists(s); },
    //       wait_timeout);
    // }

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
                // if (reflections_by_image.find(image_num)
                //     == reflections_by_image.end()) {
                //     completed_images += 1;
                //     continue;
                // }

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
