/**
 * @file spotfinder.cc
 * @brief GPU-accelerated spotfinding for crystallography diffraction images.
 *
 * This application processes crystallography diffraction images to identify
 * "strong" pixels that likely correspond to Bragg reflections. It supports
 * both rotation datasets (with 3D connected component analysis) and still
 * datasets (with 2D analysis).
 *
 * The main processing pipeline is:
 * 1. Parse arguments and initialize the data reader
 * 2. Configure detector geometry and wavelength
 * 3. Upload mask to GPU and apply resolution filtering if requested
 * 4. Spawn worker threads to process images in parallel
 * 5. For each image: decompress, copy to GPU, run spotfinding, analyze components
 * 6. For rotation datasets: combine slices into 3D reflections
 * 7. Write results to HDF5 if requested
 */
#include "spotfinder.cuh"

#include <bitshuffle.h>
#include <lodepng.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <csignal>
#include <dx2/detector.hpp>
#include <dx2/reflection.hpp>
#include <dx2/scan.hpp>
#include <iostream>
#include <memory>
#include <stop_token>
#include <thread>

#include "cbfread.hpp"
#include "common.hpp"
#include "connected_components/connected_components.hpp"
#include "cuda_common.hpp"
#include "dispersion_algorithm.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"
#include "mask_utils.hpp"
#include "pipe_handler.hpp"
#include "shmread.hpp"
#include "signal_handler.hpp"
#include "spotfinder_args.hpp"
#include "standalone.h"
#include "version.hpp"
#include "wait_utils.hpp"

using namespace std::chrono_literals;
using json = nlohmann::json;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if two floating point values are approximately equal.
 */
inline bool are_close(float a, float b, float tolerance) {
    return std::fabs(a - b) < tolerance;
}

// ============================================================================
// Helper Functions for Diagnostic Output
// ============================================================================

/**
 * @brief Write diagnostic output images for debugging spotfinding results.
 */
void write_diagnostic_images(const pixel_t *host_image,
                             const uint8_t *host_results,
                             const std::vector<Reflection> &boxes,
                             uint32_t width,
                             uint32_t height,
                             int image_num) {
    auto buffer = std::vector<std::array<uint8_t, 3>>(width * height, {0, 0, 0});
    constexpr std::array<uint8_t, 3> color_pixel{255, 0, 0};

    for (uint32_t y = 0, k = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x, ++k) {
            uint8_t graysc_value =
              std::max(0.0f, 255.99f - static_cast<float>(host_image[k]) * 10);
            buffer[k] = {graysc_value, graysc_value, graysc_value};
        }
    }

    // Draw bounding boxes
    for (size_t i = 0; i < boxes.size(); ++i) {
        auto &box = boxes[i];
        constexpr std::array<uint8_t, 3> color_shoebox{0, 0, 255};
        constexpr int edgeMin = 5, edgeMax = 7;
        for (int edge = edgeMin; edge <= edgeMax; ++edge) {
            for (int x = box.l - edge; x <= static_cast<int>(box.r) + edge; ++x) {
                buffer[width * (box.t - edge) + x] = color_shoebox;
                buffer[width * (box.b + edge) + x] = color_shoebox;
            }
            for (int y = box.t - edge; y <= static_cast<int>(box.b) + edge; ++y) {
                buffer[width * y + box.l - edge] = color_shoebox;
                buffer[width * y + box.r + edge] = color_shoebox;
            }
        }
    }

    // Draw strong pixels on top
    for (uint32_t y = 0, k = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x, ++k) {
            if (host_results[k]) {
                buffer[k] = color_pixel;
            }
        }
    }

    lodepng::encode(fmt::format("image_{:05d}.png", image_num),
                    reinterpret_cast<uint8_t *>(buffer.data()),
                    width,
                    height,
                    LCT_RGB);

    auto out = fmt::output_file(fmt::format("pixels_{:05d}.txt", image_num));
    for (uint32_t y = 0, k = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x, ++k) {
            if (host_results[k]) {
                out.print("{:4d}, {:4d}\n", x, y);
            }
        }
    }
}

/**
 * @brief Write mask as a diagnostic PNG image.
 */
void write_mask_diagnostic(const uint8_t *mask_data,
                           uint32_t width,
                           uint32_t height,
                           const std::string &filename) {
    auto image_mask = std::vector<std::array<uint8_t, 3>>(width * height, {0, 0, 0});
    for (uint32_t y = 0, k = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x, ++k) {
            image_mask[k] = {255, 255, 255};
            if (!mask_data[k]) {
                image_mask[k] = {255, 0, 0};
            }
        }
    }
    lodepng::encode(filename,
                    reinterpret_cast<uint8_t *>(image_mask.data()),
                    width,
                    height,
                    LCT_RGB);
}

// ============================================================================
// Results Processing Functions
// ============================================================================

/**
 * @brief Calculate spot variances for 3D reflections in Kabsch space.
 */
void calculate_spot_variances(const std::vector<Reflection3D> &reflections_3d,
                              const Panel &panel,
                              const Scan &scan,
                              const Vector3d &s0,
                              const Vector3d &m2,
                              std::vector<double> &sigma_b_variances,
                              std::vector<double> &sigma_m_variances,
                              std::vector<int> &bbox_depths) {
    constexpr double deg_to_rad = M_PI / 180.0;
    constexpr double rad_to_deg = 180.0 / M_PI;
    int image_range_0 = scan.get_image_range()[0];
    double oscillation_width = scan.get_oscillation()[1];
    double oscillation_start = scan.get_oscillation()[0];

    sigma_b_variances.reserve(reflections_3d.size());
    sigma_m_variances.reserve(reflections_3d.size());
    bbox_depths.reserve(reflections_3d.size());

    double sum_sigma_b_variance = 0.0;
    double sum_sigma_m_variance = 0.0;
    constexpr int min_bbox_depth = 5;
    int n_sigma_m = 0;

    for (const auto &refl : reflections_3d) {
        auto [x, y, z] = refl.center_of_mass();
        auto [xmm, ymm] = panel.px_to_mm(x, y);
        Vector3d s1 = panel.get_lab_coord(xmm, ymm);
        double phi =
          (oscillation_start + (z - image_range_0) * oscillation_width) * deg_to_rad;
        auto [sigma_b_variance, sigma_m_variance, bbox_depth] =
          refl.variances_in_kabsch_space(s1, s0, m2, panel, scan, phi);
        sigma_b_variances.push_back(sigma_b_variance);
        sigma_m_variances.push_back(sigma_m_variance);
        bbox_depths.push_back(bbox_depth);
        sum_sigma_b_variance += sigma_b_variance;
        if (bbox_depth >= min_bbox_depth) {
            sum_sigma_m_variance += sigma_m_variance;
            n_sigma_m++;
        }
    }

    // Print estimated average values
    if (reflections_3d.size()) {
        double est_sigma_b =
          std::sqrt(sum_sigma_b_variance / reflections_3d.size()) * rad_to_deg;
        logger.info("Estimated sigma_b (degrees): {:.6f}", est_sigma_b);
    }
    if (n_sigma_m) {
        double est_sigma_m = std::sqrt(sum_sigma_m_variance / n_sigma_m) * rad_to_deg;
        logger.info("Estimated sigma_m (degrees): {:.6f}, calculated on {} spots",
                    est_sigma_m,
                    n_sigma_m);
    }
}

/**
 * @brief Write 3D reflections debug output to text file.
 */
void write_3d_reflections_debug(const std::vector<Reflection3D> &reflections_3d) {
    std::ofstream out("3d_reflections.txt");
    for (const auto &reflection : reflections_3d) {
        auto [x, y, z] = reflection.center_of_mass();
        std::string reflection_info =
          fmt::format("X: [{}, {}] Y: [{}, {}] Z: [{}, {}] COM: ({}, {}, {})",
                      reflection.get_x_min(),
                      reflection.get_x_max(),
                      reflection.get_y_min(),
                      reflection.get_y_max(),
                      reflection.get_z_min(),
                      reflection.get_z_max(),
                      x,
                      y,
                      z);
        logger.trace(reflection_info);
        out << "X: [" << reflection.get_x_min() << ", " << reflection.get_x_max()
            << "] ";
        out << "Y: [" << reflection.get_y_min() << ", " << reflection.get_y_max()
            << "] ";
        out << "Z: [" << reflection.get_z_min() << ", " << reflection.get_z_max()
            << "] ";
        out << "COM: (" << x << ", " << y << ", " << z << ")\n";
    }
    logger.flush();
}

/**
 * @brief Write 3D reflections to HDF5 file.
 */
void write_3d_reflections_to_h5(const std::vector<Reflection3D> &reflections_3d,
                                const std::vector<double> &sigma_b_variances,
                                const std::vector<double> &sigma_m_variances,
                                const std::vector<int> &bbox_depths) {
    logger.debug("Writing 3D reflections to HDF5 file");

    try {
        std::vector<double> flat_coms;
        flat_coms.reserve(reflections_3d.size() * 3);

        for (const auto &refl : reflections_3d) {
            auto [x, y, z] = refl.center_of_mass();
            flat_coms.push_back(x);
            flat_coms.push_back(y);
            flat_coms.push_back(z);
        }

        ReflectionTable table;
        table.add_column("xyzobs.px.value", reflections_3d.size(), 3, flat_coms);
        std::vector<int> id(reflections_3d.size(), table.get_experiment_ids()[0]);
        table.add_column("id", reflections_3d.size(), 1, id);
        table.add_column(
          "sigma_b_variance", sigma_b_variances.size(), 1, sigma_b_variances);
        table.add_column(
          "sigma_m_variance", sigma_m_variances.size(), 1, sigma_m_variances);
        table.add_column("spot_extent_z", bbox_depths.size(), 1, bbox_depths);

        table.write("results_ffs.h5", "dials/processing/group_0");
        logger.info("Successfully wrote 3D reflections to HDF5 file");
    } catch (const std::exception &e) {
        logger.error("Error writing data to HDF5 file: {}", e.what());
    } catch (...) {
        logger.error("Unknown error writing data to HDF5 file");
    }
}

/**
 * @brief Write 2D reflections to HDF5 file.
 */
void write_2d_reflections_to_h5(
  const std::map<int, std::vector<float>> &reflection_centers_2d) {
    logger.info("Processing 2D spots");
    logger.debug("Writing 2D reflections to HDF5 file");

    try {
        std::vector<double> flat_coms;
        std::vector<int> ids;
        std::vector<int> centers_map_keys;
        for (const auto &pair : reflection_centers_2d) {
            centers_map_keys.push_back(pair.first);
        }
        std::sort(centers_map_keys.begin(), centers_map_keys.end());
        int id = 0;
        for (int imageno : centers_map_keys) {
            std::vector<float> flat_coms_this = reflection_centers_2d.at(imageno);
            int n_refls = flat_coms_this.size() / 3;
            for (auto com : flat_coms_this) {
                flat_coms.push_back(static_cast<double>(com));
            }
            for (int i = 0; i < n_refls; ++i) {
                ids.push_back(id);
            }
            id += 1;
        }

        ReflectionTable table;
        for (int i = 0; i < id - 1; ++i) {
            table.generate_new_attributes();
        }
        table.add_column("xyzobs.px.value", flat_coms.size() / 3, 3, flat_coms);
        table.add_column("id", ids.size(), 1, ids);

        table.write("results_ffs.h5", "dials/processing/group_0");
        logger.info("Successfully wrote {} 2D reflections to HDF5 file", ids.size());
    } catch (const std::exception &e) {
        logger.error("Error writing data to HDF5 file: {}", e.what());
    } catch (...) {
        logger.error("Unknown error writing data to HDF5 file");
    }
    logger.info("2D spot analysis complete");
}

/**
 * @brief Process 3D connected components and write results.
 */
void process_3d_results(
  std::map<int, std::unique_ptr<ConnectedComponents>> &rotation_slices,
  uint32_t width,
  uint32_t height,
  uint32_t num_images,
  uint32_t min_spot_size_3d,
  float max_peak_centroid_separation,
  const detector_geometry &detector,
  float wavelength,
  float oscillation_start,
  float oscillation_width,
  bool save_to_h5,
  bool do_writeout) {
    logger.info("Processing 3D spots");

    // Step 1: Convert rotation_slices map to a vector
    std::vector<std::unique_ptr<ConnectedComponents>> slices;
    for (auto &[image_num, connected_components] : rotation_slices) {
        slices.push_back(std::move(connected_components));
    }

    // Step 2: Find 3D connected components
    auto reflections_3d = ConnectedComponents::find_3d_components(
      slices, width, height, min_spot_size_3d, max_peak_centroid_separation);

    // Step 3: Output the 3D reflections
    logger.info(
      fmt::format("Found {} spots", fmt::styled(reflections_3d.size(), fmt_cyan)));

    if (do_writeout) {
        write_3d_reflections_debug(reflections_3d);
    }

    // Step 4: Calculate spot variances
    std::array<int, 2> image_size = {static_cast<int>(width), static_cast<int>(height)};
    Panel panel(detector.distance * 1000,
                {detector.beam_center_x, detector.beam_center_y},
                {detector.pixel_size_x * 1000, detector.pixel_size_y * 1000},
                image_size);
    Vector3d s0 = {0.0, 0.0, -1.0 / wavelength};
    Scan scan({1, static_cast<int>(num_images)},
              {oscillation_start, oscillation_width});
    Vector3d m2 = {1.0, 0.0, 0.0};  // Rotation axis, assumed to be +x

    std::vector<double> sigma_b_variances;
    std::vector<double> sigma_m_variances;
    std::vector<int> bbox_depths;

    calculate_spot_variances(reflections_3d,
                             panel,
                             scan,
                             s0,
                             m2,
                             sigma_b_variances,
                             sigma_m_variances,
                             bbox_depths);

    // Step 5: Write to HDF5 if requested
    if (save_to_h5) {
        write_3d_reflections_to_h5(
          reflections_3d, sigma_b_variances, sigma_m_variances, bbox_depths);
    }

    logger.info("3D spot analysis complete");
}

// ============================================================================
// Main Application Entry Point
// ============================================================================

int main(int argc, char **argv) {
    logger.info("Spotfinder version: {}", FFS_VERSION);

    // =========================================================================
    // Step 1: Parse command-line arguments
    // =========================================================================
    SpotfinderArgumentParser parser(FFS_VERSION);
    auto args = parser.parse_args(argc, argv);

    auto const file = parser.file();
    bool do_validate = parser.get<bool>("validate");
    bool do_writeout = parser.get<bool>("writeout");
    int pipe_fd = parser.get<int>("pipe_fd");
    float wait_timeout = parser.get<float>("timeout");
    bool save_to_h5 = parser.get<bool>("save-h5");
    bool output_for_index = parser.get<bool>("output-for-index");
    float dmin = parser.get<float>("dmin");
    float dmax = parser.get<float>("dmax");

    DispersionAlgorithm dispersion_algorithm(parser.get<std::string>("algorithm"));
    fmt::print("Algorithm: {}\n",
               fmt::styled(dispersion_algorithm.algorithm_str, fmt_green));

    uint32_t num_cpu_threads = parser.get<uint32_t>("threads");
    if (num_cpu_threads < 1) {
        fmt::print("Error: Thread count must be >= 1\n");
        std::exit(1);
    }

    uint32_t min_spot_size = parser.get<uint32_t>("min-spot-size");
    uint32_t min_spot_size_3d = parser.get<uint32_t>("min-spot-size-3d");
    float max_peak_centroid_separation =
      parser.get<float>("max-peak-centroid-separation");

    // =========================================================================
    // Step 2: Initialize the data reader
    // =========================================================================
    std::unique_ptr<Reader> reader_ptr;

    // Wait for file to exist
    if (!std::filesystem::exists(file)) {
        wait_for_ready_for_read(
          file,
          [](const std::string &s) { return std::filesystem::exists(s); },
          wait_timeout);
    }

    // Create appropriate reader based on file type
    if (std::filesystem::is_directory(file)) {
        wait_for_ready_for_read(file, is_ready_for_read<SHMRead>, wait_timeout);
        reader_ptr = std::make_unique<SHMRead>(file);
    } else if (file.ends_with(".cbf")) {
        if (!parser.is_used("images")) {
            fmt::print("Error: CBF reading must specify --images\n");
            std::exit(1);
        }
        reader_ptr = std::make_unique<CBFRead>(
          file, parser.get<uint32_t>("images"), parser.get<uint32_t>("start-index"));
    } else {
        wait_for_ready_for_read(file, is_ready_for_read<H5Read>, wait_timeout);
        reader_ptr =
          file.empty() ? std::make_unique<H5Read>() : std::make_unique<H5Read>(file);
    }

    Reader &reader = *reader_ptr;
    auto reader_mutex = std::mutex{};

    uint32_t num_images = parser.is_used("images") ? parser.get<uint32_t>("images")
                                                   : reader.get_number_of_images();
    uint32_t height = reader.image_shape()[0];
    uint32_t width = reader.image_shape()[1];
    auto trusted_px_max = reader.get_trusted_range()[1];

    // =========================================================================
    // Step 3: Configure detector geometry and wavelength
    // =========================================================================
    detector_geometry detector;

    if (parser.is_used("detector")) {
        std::string detector_json = parser.get<std::string>("detector");
        json detector_json_obj = json::parse(detector_json);
        detector = detector_geometry(detector_json_obj);

        if (do_validate) {
            auto beam_center = reader.get_beam_center();
            auto pixel_size = reader.get_pixel_size();
            auto distance = reader.get_detector_distance();
            if (beam_center
                && (!are_close(detector.beam_center_x, beam_center.value()[1], 0.1)
                    || !are_close(
                      detector.beam_center_y, beam_center.value()[0], 0.1))) {
                fmt::print(
                  "Warning: Beam center mismatched:\n    json:   {} px, {} px (used)\n "
                  "   reader: {} px, {} px\n",
                  detector.beam_center_x,
                  detector.beam_center_y,
                  beam_center.value()[1],
                  beam_center.value()[0]);
            }
            if (pixel_size
                && (!are_close(detector.pixel_size_x, pixel_size.value()[1], 1e-9)
                    || !are_close(
                      detector.pixel_size_y, pixel_size.value()[0], 1e-9))) {
                fmt::print(
                  "Warning: Pixel size mismatched:\n    json:   {} µm, {} µm (used)\n  "
                  "  reader: {} µm, {} µm\n",
                  detector.pixel_size_x,
                  detector.pixel_size_y,
                  pixel_size.value()[1] * 1e6,
                  pixel_size.value()[0] * 1e6);
            }
            if (distance && !are_close(distance.value(), detector.distance, 0.1e-6)) {
                fmt::print(
                  "Warning: Detector distance mismatched:\n    json:   {} m (used)\n   "
                  " reader: {} m\n",
                  detector.distance,
                  distance.value());
            }
        }
    } else {
        auto beam_center = reader.get_beam_center();
        auto pixel_size = reader.get_pixel_size();
        auto distance = reader.get_detector_distance();
        if (!beam_center) {
            fmt::print(
              "Error: No beam center available from file. Please pass detector "
              "metadata with --distance.\n");
            std::exit(1);
        }
        if (!pixel_size) {
            fmt::print(
              "Error: No pixel size available from file. Please pass detector metadata "
              "with --distance.\n");
            std::exit(1);
        }
        if (!distance) {
            fmt::print(
              "Error: No detector distance available from file. Please pass metadata "
              "with --distance.\n");
            std::exit(1);
        }
        detector =
          detector_geometry(distance.value(), beam_center.value(), pixel_size.value());
    }

    float wavelength;
    if (parser.is_used("wavelength")) {
        wavelength = parser.get<float>("wavelength");
        if (do_validate && reader.get_wavelength()
            && reader.get_wavelength().value() != wavelength) {
            fmt::print(
              "Warning: Wavelength mismatch:\n    Argument: {} Å\n    Reader:   {} Å\n",
              wavelength,
              reader.get_wavelength().value());
        }
    } else {
        auto wavelength_opt = reader.get_wavelength();
        if (!wavelength_opt) {
            fmt::print(
              "Error: No wavelength provided. Please pass wavelength using: "
              "--wavelength\n");
            std::exit(1);
        }
        wavelength = wavelength_opt.value();
        printf("Got wavelength from file: %f Å\n", wavelength);
    }

    fmt::print(
      "Detector geometry:\n"
      "    Distance:    {0:.1f} mm\n"
      "    Beam Center: {1:.1f} px {2:.1f} px\n"
      "Beam Wavelength: {3:.2f} Å\n",
      fmt::styled(detector.distance * 1000, fmt_cyan),
      fmt::styled(detector.beam_center_x, fmt_cyan),
      fmt::styled(detector.beam_center_y, fmt_cyan),
      fmt::styled(wavelength, fmt_cyan));

    auto [oscillation_start, oscillation_width] = reader.get_oscillation();
    if (oscillation_width > 0) {
        fmt::print("Oscillation:  Start: {:.2f}°  Width: {:.2f}°\n",
                   fmt::styled(oscillation_start, fmt_cyan),
                   fmt::styled(oscillation_width, fmt_cyan));
    }

    // =========================================================================
    // Step 4: Set up signal handling
    // =========================================================================
    std::signal(SIGINT, stop_processing);

    // =========================================================================
    // Step 5: Configure GPU execution parameters
    // =========================================================================
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

    // =========================================================================
    // Step 6: Upload mask and apply resolution filtering
    // =========================================================================
    auto mask = upload_mask(reader);

    if (do_writeout && reader.get_mask()) {
        write_mask_diagnostic(
          reader.get_mask()->data(), width, height, "mask_source.png");
    }

    if (dmin > 0 || dmax > 0) {
        apply_resolution_filtering(
          mask, width, height, wavelength, detector, dmin, dmax);

        if (do_writeout) {
            auto calculated_mask = std::vector<uint8_t>(width * height, 0);
            cudaMemcpy2D(calculated_mask.data(),
                         width,
                         mask.get(),
                         mask.pitch_bytes(),
                         width,
                         height,
                         cudaMemcpyDeviceToHost);
            write_mask_diagnostic(
              calculated_mask.data(), width, height, "mask_calculated.png");
        }
    }

    // =========================================================================
    // Step 7: Initialize data structures for results collection
    // =========================================================================
    auto all_images_start_time = std::chrono::high_resolution_clock::now();
    auto next_image = std::atomic<int>(0);
    auto completed_images = std::atomic<int>(0);
    auto cpu_sync = std::barrier{num_cpu_threads};
    double time_waiting_for_images = 0.0;

    std::unique_ptr<PipeHandler> pipeHandler = nullptr;
    if (pipe_fd != -1) {
        pipeHandler = std::make_unique<PipeHandler>(pipe_fd);
    }

    std::unique_ptr<std::map<int, std::unique_ptr<ConnectedComponents>>> rotation_slices =
      nullptr;
    std::mutex rotation_slices_mutex;
    std::unique_ptr<std::map<int, std::vector<float>>> reflection_centers_2d = nullptr;
    std::mutex reflection_centers_2d_mutex;

    if (oscillation_width > 0) {
        rotation_slices =
          std::make_unique<std::map<int, std::unique_ptr<ConnectedComponents>>>();
        fmt::print("Dataset type: {}\n", fmt::styled("Rotation set", fmt_magenta));
    } else {
        fmt::print("Dataset type: {}\n", fmt::styled("Still set", fmt_magenta));
        if (save_to_h5) {
            reflection_centers_2d = std::make_unique<std::map<int, std::vector<float>>>();
        }
    }

    // =========================================================================
    // Step 8: Spawn worker threads for image processing
    // =========================================================================
    std::vector<std::jthread> threads;
    for (uint32_t thread_id = 0; thread_id < num_cpu_threads; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            auto stop_token = global_stop.get_token();
            CudaStream stream;

            // Allocate thread-local buffers
            auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height);
            auto host_results = make_cuda_pinned_malloc<uint8_t>(width * height);
            auto device_image = PitchedMalloc<pixel_t>(width, height);
            auto device_results =
              PitchedMalloc<uint8_t>(make_cuda_malloc<uint8_t[]>(mask.pitch * height),
                                     width,
                                     height,
                                     mask.pitch);

            auto raw_chunk_buffer =
              std::vector<uint8_t>(width * height * sizeof(pixel_t));

            cpu_sync.arrive_and_wait();
            CudaEvent start, copy, post, postcopy, end;
            auto last_image_received = std::chrono::high_resolution_clock::now();

            // Main image processing loop
            while (!stop_token.stop_requested()) {
                auto image_num = next_image.fetch_add(1);
                if (image_num >= num_images) {
                    break;
                }
                auto offset_image_num = image_num + parser.get<uint32_t>("start-index");

                // Wait for image availability
                {
                    std::scoped_lock lock(reader_mutex);
                    auto swmr_wait_start_time =
                      std::chrono::high_resolution_clock::now();

                    while (!reader.is_image_available(offset_image_num)
                           && !stop_token.stop_requested()) {
                        auto current_time = std::chrono::high_resolution_clock::now();
                        auto elapsed_wait_time =
                          std::chrono::duration_cast<std::chrono::duration<double>>(
                            current_time - last_image_received)
                            .count();

                        if (elapsed_wait_time > wait_timeout) {
                            fmt::print("Timeout waiting for image {}\n",
                                       offset_image_num);
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

                // Fetch raw image data
                std::span<uint8_t> buffer;
                while (true) {
                    {
                        std::scoped_lock lock(reader_mutex);
                        buffer =
                          reader.get_raw_chunk(offset_image_num, raw_chunk_buffer);
                    }
                    if (buffer.size() == 0) {
                        fmt::print(fmt::runtime(
                          "\033[1mRace Condition?!?? Got buffer size 0 for image "
                          "{image_num}. Sleeping.\033[0m\n"));
                        std::this_thread::sleep_for(100ms);
                        continue;
                    }
                    break;
                }

                // Decompress image data
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

                // Copy image to GPU
                start.record(stream);
                CUDA_CHECK(cudaMemcpy2DAsync(device_image.get(),
                                             device_image.pitch_bytes(),
                                             host_image.get(),
                                             width * sizeof(pixel_t),
                                             width * sizeof(pixel_t),
                                             height,
                                             cudaMemcpyHostToDevice,
                                             stream));
                copy.record(stream);

                // Run spotfinding kernel
                switch (dispersion_algorithm.algorithm) {
                case DispersionAlgorithm::Algorithm::DISPERSION:
                    call_do_spotfinding_dispersion(blocks_dims,
                                                   gpu_thread_block_size,
                                                   stream,
                                                   device_image,
                                                   mask,
                                                   width,
                                                   height,
                                                   trusted_px_max,
                                                   &device_results);
                    break;
                case DispersionAlgorithm::Algorithm::DISPERSION_EXTENDED:
                    call_do_spotfinding_extended(blocks_dims,
                                                 gpu_thread_block_size,
                                                 stream,
                                                 device_image,
                                                 mask,
                                                 width,
                                                 height,
                                                 trusted_px_max,
                                                 &device_results,
                                                 do_writeout);
                    break;
                }
                post.record(stream);

                // Copy results back to CPU
                CUDA_CHECK(cudaMemcpy2DAsync(host_results.get(),
                                             width * sizeof(uint8_t),
                                             device_results.get(),
                                             device_results.pitch_bytes(),
                                             width * sizeof(uint8_t),
                                             height,
                                             cudaMemcpyDeviceToHost,
                                             stream));
                postcopy.record(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                // Run connected components analysis
                std::unique_ptr<ConnectedComponents> connected_components_2d =
                  std::make_unique<ConnectedComponents>(
                    host_results.get(), host_image.get(), width, height, min_spot_size);

                auto boxes = connected_components_2d->get_boxes();
                size_t num_strong_pixels =
                  connected_components_2d->get_num_strong_pixels();
                size_t num_strong_pixels_filtered =
                  connected_components_2d->get_num_strong_pixels_filtered();

                std::vector<float> centers_of_mass;

                // Store results based on dataset type
                if (oscillation_width > 0) {
                    std::lock_guard<std::mutex> lock(rotation_slices_mutex);
                    (*rotation_slices)[offset_image_num] =
                      std::move(connected_components_2d);
                } else if (save_to_h5 || output_for_index) {
                    std::vector<Reflection3D> reflections =
                      connected_components_2d->find_2d_components(
                        min_spot_size, max_peak_centroid_separation);
                    for (const auto &r : reflections) {
                        auto [x, y, z] = r.center_of_mass();
                        centers_of_mass.push_back(x);
                        centers_of_mass.push_back(y);
                        centers_of_mass.push_back(z);
                    }
                    if (save_to_h5) {
                        std::lock_guard<std::mutex> lock(reflection_centers_2d_mutex);
                        (*reflection_centers_2d)[offset_image_num] = centers_of_mass;
                    }
                }

                end.record(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                // Write diagnostic output if requested
                if (do_writeout) {
                    write_diagnostic_images(host_image.get(),
                                            host_results.get(),
                                            boxes,
                                            width,
                                            height,
                                            image_num);
                }

                // Send data through pipe if configured
                if (pipeHandler != nullptr) {
                    json json_data = {{"num_strong_pixels", num_strong_pixels},
                                      {"file", file},
                                      {"file-number", image_num},
                                      {"n_spots_total", boxes.size()}};
                    if (output_for_index) {
                        json_data["spot_centers"] = centers_of_mass;
                    }
                    pipeHandler->sendData(json_data);
                }

                // Validation or logging
                if (do_validate) {
                    size_t count = 0;
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            if (host_results[x + width * y]) {
                                ++count;
                            }
                        }
                    }
                    auto spotfinder = StandaloneSpotfinder(width, height);
                    auto converted_image = std::vector<double>{
                      host_image.get(), host_image.get() + width * height};
                    auto dials_strong = spotfinder.standard_dispersion(
                      converted_image,
                      reader.get_mask().value_or(std::span<uint8_t>{}));
                    size_t mismatch_x = 0, mismatch_y = 0;
                    bool validation_matches = compare_results(dials_strong.data(),
                                                              width,
                                                              host_results.get(),
                                                              width,
                                                              width,
                                                              height,
                                                              &mismatch_x,
                                                              &mismatch_y);
                    if (validation_matches) {
                        fmt::print(
                          "Thread {:2d}, Image {:4d}: Compared: \033[32mMatch {} "
                          "px\033[0m\n",
                          thread_id,
                          image_num,
                          count);
                    } else {
                        fmt::print(
                          "Thread {:2d}, Image {:4d}: Compared: "
                          "\033[1;31mMismatch ({} px from kernel)\033[0m\n",
                          thread_id,
                          image_num,
                          count);
                    }
                } else {
                    if (num_cpu_threads == 1) {
                        fmt::print(
                          "Thread {:2d} finished image {:4d}\n"
                          "       Copy: {:5.1f} ms\n"
                          "     Kernel: {:5.1f} ms\n"
                          "  Post Copy: {:5.1f} ms\n"
                          "       Post: {:5.1f} ms\n"
                          "             ════════\n"
                          "     Total:  {:5.1f} ms ({:.1f} GBps)\n"
                          "    {} strong pixels\n"
                          "    {} filtered reflections ({} pixels)\n",
                          thread_id,
                          image_num,
                          copy.elapsed_time(start),
                          post.elapsed_time(start),
                          postcopy.elapsed_time(post),
                          end.elapsed_time(postcopy),
                          end.elapsed_time(start),
                          GBps<pixel_t>(end.elapsed_time(start), width * height),
                          bold(num_strong_pixels),
                          bold(boxes.size()),
                          bold(num_strong_pixels_filtered));
                    } else {
                        fmt::print(
                          "Thread {:2d} finished image {:4d} with {:5d} strong pixels, "
                          "{:4d} filtered reflections ({} pixels)\n",
                          thread_id,
                          image_num,
                          num_strong_pixels,
                          boxes.size(),
                          num_strong_pixels_filtered);
                    }
                }
                completed_images += 1;
            }
        });
    }

    // Wait for all threads to finish
    for (auto &thread : threads) {
        thread.join();
    }

    // =========================================================================
    // Step 9: Process and write results
    // =========================================================================
    if (oscillation_width > 0) {
        process_3d_results(*rotation_slices,
                           width,
                           height,
                           num_images,
                           min_spot_size_3d,
                           max_peak_centroid_separation,
                           detector,
                           wavelength,
                           oscillation_start,
                           oscillation_width,
                           save_to_h5,
                           do_writeout);
    } else if (save_to_h5) {
        write_2d_reflections_to_h5(*reflection_centers_2d);
    }

    // =========================================================================
    // Step 10: Print summary statistics
    // =========================================================================
    float total_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - all_images_start_time)
        .count();
    fmt::print(
      "\n{} images in {:.2f} s (\033[1;34m{:.2f} GBps\033[0m) (\033[1;34m{:.1f} "
      "fps\033[0m)\n",
      int(completed_images),
      total_time,
      GBps<pixel_t>(
        total_time * 1000,
        static_cast<size_t>(width) * static_cast<size_t>(height) * completed_images),
      completed_images / total_time,
      width,
      height);
    if (time_waiting_for_images < 10) {
        fmt::print("Total time waiting for images to appear: {:.0f} ms\n",
                   time_waiting_for_images * 1000);
    } else {
        fmt::print("Total time waiting for images to appear: {:.2f} s\n",
                   time_waiting_for_images);
    }

    return 0;
}
