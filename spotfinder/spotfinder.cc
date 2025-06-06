#include "spotfinder.cuh"

#include <bitshuffle.h>
#include <lodepng.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <dx2/reflection.hpp>
#include <iostream>
#include <memory>
#include <ranges>
#include <stop_token>
#include <thread>
#include <utility>

#include "cbfread.hpp"
#include "common.hpp"
#include "connected_components/connected_components.hpp"
#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"
#include "kernels/masking.cuh"
#include "shmread.hpp"
#include "standalone.h"
#include "version.hpp"

using namespace std::chrono_literals;
using json = nlohmann::json;

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

bool are_close(float a, float b, float tolerance) {
    return std::fabs(a - b) < tolerance;
}

/// Copy the mask from a reader into a pitched GPU area
template <typename T>
auto upload_mask(T &reader) -> PitchedMalloc<uint8_t> {
    size_t height = reader.image_shape()[0];
    size_t width = reader.image_shape()[1];

    auto [dev_mask, device_mask_pitch] =
      make_cuda_pitched_malloc<uint8_t>(width, height);

    size_t valid_pixels = 0;
    CudaEvent start, end;
    if (reader.get_mask()) {
        // Count how many valid Mpx in this mask
        for (size_t i = 0; i < width * height; ++i) {
            if (reader.get_mask().value()[i]) {
                valid_pixels += 1;
            }
        }
        start.record();
        cudaMemcpy2DAsync(dev_mask.get(),
                          device_mask_pitch,
                          reader.get_mask()->data(),
                          width,
                          width,
                          height,
                          cudaMemcpyHostToDevice);
        cuda_throw_error();
    } else {
        valid_pixels = width * height;
        start.record();
        cudaMemset(dev_mask.get(), 1, device_mask_pitch * height);
        cuda_throw_error();
    }
    end.record();
    end.synchronize();

    float memcpy_time = end.elapsed_time(start);
    fmt::print("Uploaded mask ({:.2f} Mpx) in {:.2f} ms ({:.1f} GBps)\n",
               static_cast<float>(valid_pixels) / 1e6,
               memcpy_time,
               GBps(memcpy_time, width * height));

    return PitchedMalloc{
      dev_mask,
      width,
      height,
      device_mask_pitch,
    };
}

void apply_resolution_filtering(PitchedMalloc<uint8_t> mask,
                                int width,
                                int height,
                                float wavelength,
                                detector_geometry detector,
                                float dmin,
                                float dmax,
                                cudaStream_t stream = 0) {
    // Define the block size and grid size for the kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Set the parameters for the resolution mask kernel
    ResolutionMaskParams params{.mask_pitch = mask.pitch,
                                .width = width,
                                .height = height,
                                .wavelength = wavelength,
                                .detector = detector,
                                .dmin = dmin,
                                .dmax = dmax};

    // Launch the kernel to apply resolution filtering
    call_apply_resolution_mask(
      numBlocks, threadsPerBlock, 0, stream, mask.get(), params);

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

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
            fmt::print("\r{}  {} [{:4.1f} s] ", message_prefix, ball[i], wait_time);
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
 * @brief Struct to store the dispersion algorithm and its string representation.
 */
struct DispersionAlgorithm {
    std::string algorithm_str;
    enum class Algorithm { DISPERSION, DISPERSION_EXTENDED };
    Algorithm algorithm;

    /**
     * @brief Constructor to initialize the DispersionAlgorithm object.
     * @param input The string representation of the algorithm.
     */
    DispersionAlgorithm(std::string input) {
        // Convert the input to lowercase for case-insensitive comparison
        this->algorithm_str = input;
        std::transform(input.begin(), input.end(), input.begin(), ::tolower);
        if (input == "dispersion") {
            this->algorithm_str = "Dispersion";
            this->algorithm = Algorithm::DISPERSION;
        } else if (input == "dispersion_extended") {
            this->algorithm_str = "Dispersion Extended";  // ✨
            this->algorithm = Algorithm::DISPERSION_EXTENDED;
        } else {
            throw std::invalid_argument("Invalid algorithm specified");
        }
    }
};

/**
 * @brief Class for handling a pipe and sending data through it in a thread-safe manner.
 */
class PipeHandler {
  private:
    int pipe_fd;     // File descriptor for the pipe
    std::mutex mtx;  // Mutex for synchronization

  public:
    /**
     * @brief Constructor to initialize the PipeHandler object.
     * @param pipe_fd The file descriptor for the pipe.
     */
    PipeHandler(int pipe_fd) : pipe_fd(pipe_fd) {
        // Constructor to initialize the pipe handler
        fmt::print("PipeHandler initialized with pipe_fd: {}\n", pipe_fd);
    }

    /**
     * @brief Destructor to close the pipe.
     */
    ~PipeHandler() {
        close(pipe_fd);
    }

    /**
     * @brief Sends data through the pipe in a thread-safe manner.
     * @param json_data A json object containing the data to be sent.
     */
    void sendData(const json &json_data) {
        // Lock the mutex, to ensure that only one thread writes to the pipe at a time
        // This unlocks the mutex when the function returns
        std::lock_guard<std::mutex> lock(mtx);

        // Convert the JSON object to a string
        std::string stringified_json = json_data.dump() + "\n";

        // Write the data to the pipe
        // Returns the number of bytes written to the pipe
        // Returns -1 if an error occurs
        ssize_t bytes_written =
          write(pipe_fd, stringified_json.c_str(), stringified_json.length());

        // Check if an error occurred while writing to the pipe
        if (bytes_written == -1) {
            std::cerr << "Error writing to pipe: " << strerror(errno) << std::endl;
        } else {
            // fmt::print("Data sent through the pipe: {}\n", stringified_json);
        }
    }
};

int main(int argc, char **argv) {
    logger.info("Spotfinder version: {}", FFS_VERSION);

#pragma region Argument Parsing
    // Parse arguments and get our H5Reader
    auto parser = CUDAArgumentParser(FFS_VERSION);
    parser.add_h5read_arguments();
    parser.add_argument("-n", "--threads")
      .help("Number of parallel reader threads")
      .default_value<uint32_t>(1)
      .metavar("NUM")
      .scan<'u', uint32_t>();
    parser.add_argument("--validate")
      .help("Run DIALS standalone validation")
      .default_value(false)
      .implicit_value(true);
    parser.add_argument("--images")
      .help("Maximum number of images to process")
      .metavar("NUM")
      .scan<'u', uint32_t>();
    parser.add_argument("--writeout")
      .help("Write diagnostic output images")
      .default_value(false)
      .implicit_value(true);
    parser.add_argument("--min-spot-size")
      .help("2D Reflections with a pixel count below this will be discarded.")
      .metavar("N")
      .default_value<uint32_t>(3)
      .scan<'u', uint32_t>();
    parser.add_argument("--min-spot-size-3d")
      .help("3D Reflections with a pixel count below this will be discarded.")
      .metavar("N")
      .default_value<uint32_t>(3)
      .scan<'u', uint32_t>();
    parser.add_argument("--start-index")
      .help("Index of first image. Only used for CBF reading, and can only be 0 or 1.")
      .metavar("N")
      .default_value<uint32_t>(0)
      .scan<'u', uint32_t>();
    parser.add_argument("-t", "--timeout")
      .help("Amount of time (in seconds) to wait for new images before failing.")
      .metavar("S")
      .default_value<float>(30)
      .scan<'f', float>();
    parser.add_argument("-fd", "--pipe_fd")
      .help("File descriptor for the pipe to output data through")
      .metavar("FD")
      .default_value<int>(-1)
      .scan<'i', int>();
    parser.add_argument("-a", "--algorithm")
      .help("Dispersion algorithm to use")
      .metavar("ALGO")
      .default_value<std::string>("dispersion");
    parser.add_argument("--dmin")
      .help("Minimum resolution (Å)")
      .metavar("MIN D")
      .default_value<float>(-1.f)
      .scan<'f', float>();
    parser.add_argument("--dmax")
      .help("Maximum resolution (Å)")
      .metavar("MAX D")
      .default_value<float>(-1.f)
      .scan<'f', float>();
    parser.add_argument("-w", "-λ", "--wavelength")
      .help("Wavelength of the X-ray beam (Å)")
      .metavar("λ")
      .scan<'f', float>();
    parser.add_argument("--detector").help("Detector geometry JSON").metavar("JSON");
    parser.add_argument("-h5", "--save-h5")
      .help("Save the output to an HDF5 file")
      .metavar("FILE")
      .default_value(false)
      .implicit_value(true);

    auto args = parser.parse_args(argc, argv);
    bool do_validate = parser.get<bool>("validate");
    bool do_writeout = parser.get<bool>("writeout");
    int pipe_fd = parser.get<int>("pipe_fd");
    float wait_timeout = parser.get<float>("timeout");
    bool save_to_h5 = parser.get<bool>("save-h5");

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

    std::unique_ptr<Reader> reader_ptr;

    // Wait for read-readiness
    // Firstly: That the path exists at all
    if (!std::filesystem::exists(args.file)) {
        wait_for_ready_for_read(
          args.file,
          [](const std::string &s) { return std::filesystem::exists(s); },
          wait_timeout);
    }
    if (std::filesystem::is_directory(args.file)) {
        wait_for_ready_for_read(args.file, is_ready_for_read<SHMRead>, wait_timeout);
        reader_ptr = std::make_unique<SHMRead>(args.file);
    } else if (args.file.ends_with(".cbf")) {
        if (!parser.is_used("images")) {
            fmt::print("Error: CBF reading must specify --images\n");
            std::exit(1);
        }
        reader_ptr = std::make_unique<CBFRead>(args.file,
                                               parser.get<uint32_t>("images"),
                                               parser.get<uint32_t>("start-index"));
    } else {
        wait_for_ready_for_read(args.file, is_ready_for_read<H5Read>, wait_timeout);
        reader_ptr = args.file.empty() ? std::make_unique<H5Read>()
                                       : std::make_unique<H5Read>(args.file);
    }
    // Bind this as a reference
    Reader &reader = *reader_ptr;

    auto reader_mutex = std::mutex{};

    uint32_t num_images = parser.is_used("images") ? parser.get<uint32_t>("images")
                                                   : reader.get_number_of_images();

    ushort height = reader.image_shape()[0];
    ushort width = reader.image_shape()[1];
    auto trusted_px_max = reader.get_trusted_range()[1];

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
                  "   "
                  "reader: "
                  "{} px, {} px\n",
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
                  "  "
                  "reader: "
                  "{} µm, {} µm\n",
                  detector.pixel_size_x,
                  detector.pixel_size_y,
                  pixel_size.value()[1] * 1e6,
                  pixel_size.value()[0] * 1e6);
            }
            if (distance && !are_close(distance.value(), detector.distance, 0.1e-6)) {
                fmt::print(
                  "Warning: Detector distance mismatched:\n    json:   {} m (used)\n   "
                  " "
                  "reader: "
                  "{} m\n",
                  detector.distance,
                  distance.value());
            }
        }
    } else {
        // Try to read detector geometry out of the reader
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
              "Warning: Wavelength mismatch:\n    Argument: {} Å\n    Reader:   {} Å\n",
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
      "Beam Wavelength: {3:.2f} Å\n",
      fmt::styled(detector.distance * 1000, fmt_cyan),
      fmt::styled(detector.beam_center_x, fmt_cyan),
      fmt::styled(detector.beam_center_y, fmt_cyan),
      fmt::styled(wavelength, fmt_cyan));

    auto [oscillation_start, oscillation_width] = reader.get_oscillation();

    /*
     * If the oscillation width is greater than 0, then this is a
     * rotation dataset. Otherwise, it is a still dataset.
    */
    if (oscillation_width > 0) {
        fmt::print("Oscillation:  Start: {:.2f}°  Width: {:.2f}°\n",
                   fmt::styled(oscillation_start, fmt_cyan),
                   fmt::styled(oscillation_width, fmt_cyan));
    }
#pragma endregion Argument Parsing

    std::signal(SIGINT, stop_processing);

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

    auto mask = upload_mask(reader);

    // Create a mask image for debugging
    if (do_writeout) {
        auto image_mask =
          std::vector<std::array<uint8_t, 3>>(width * height, {0, 0, 0});

        // Write out the raw image mask
        auto image_mask_source = reader.get_mask()->data();
        for (int y = 0, k = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x, ++k) {
                image_mask[k] = {255, 255, 255};
                if (!image_mask_source[k]) {
                    image_mask[k] = {255, 0, 0};
                }
            }
        }
        lodepng::encode("mask_source.png",
                        reinterpret_cast<uint8_t *>(image_mask.data()),
                        width,
                        height,
                        LCT_RGB);
    }

#pragma region Resolution Filtering
    // If set, apply resolution filtering
    if (dmin > 0 || dmax > 0) {
        apply_resolution_filtering(
          mask, width, height, wavelength, detector, dmin, dmax);
        if (do_writeout) {
            // Copy the mask back from the GPU
            auto calculated_mask = std::vector<uint8_t>(width * height, 0);
            cudaMemcpy2D(calculated_mask.data(),
                         width,
                         mask.get(),
                         mask.pitch_bytes(),
                         width,
                         height,
                         cudaMemcpyDeviceToHost);

            auto image_mask =
              std::vector<std::array<uint8_t, 3>>(width * height, {0, 0, 0});

            for (int y = 0, k = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x, ++k) {
                    image_mask[k] = {255, 255, 255};
                    if (!calculated_mask[k]) {
                        image_mask[k] = {255, 0, 0};
                    }
                }
            }
            lodepng::encode("mask_calculated.png",
                            reinterpret_cast<uint8_t *>(image_mask.data()),
                            width,
                            height,
                            LCT_RGB);
        }
    }
#pragma endregion Resolution Filtering

    auto all_images_start_time = std::chrono::high_resolution_clock::now();

    auto next_image = std::atomic<int>(0);
    auto completed_images = std::atomic<int>(0);

    auto cpu_sync = std::barrier{num_cpu_threads};

    auto png_write_mutex = std::mutex{};

    double time_waiting_for_images = 0.0;

    // Create a PipeHandler object if the pipe file descriptor is provided
    std::unique_ptr<PipeHandler> pipeHandler = nullptr;
    if (pipe_fd != -1) {
        pipeHandler = std::make_unique<PipeHandler>(pipe_fd);
    }

    // Create a unique pointer to store the image slices if this is a rotation dataset
    std::unique_ptr<std::unordered_map<int, std::unique_ptr<ConnectedComponents>>>
      rotation_slices = nullptr;
    std::mutex rotation_slices_mutex;  // Mutex to protect the rotation slices map
    if (oscillation_width > 0) {
        // If oscillation information is available then this is a rotation dataset
        fmt::print("Dataset type: {}\n", fmt::styled("Rotation set", fmt_magenta));
        rotation_slices = std::make_unique<
          std::unordered_map<int, std::unique_ptr<ConnectedComponents>>>();
    } else {
        fmt::print("Dataset type: {}\n", fmt::styled("Still set", fmt_magenta));
    }

    // Spawn the reader threads
    std::vector<std::jthread> threads;
    for (int thread_id = 0; thread_id < num_cpu_threads; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            auto stop_token = global_stop.get_token();
            CudaStream stream;

            auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height);
            auto host_results = make_cuda_pinned_malloc<uint8_t>(width * height);
            auto device_image = PitchedMalloc<pixel_t>(width, height);
            auto device_results =
              PitchedMalloc<uint8_t>(make_cuda_malloc<uint8_t[]>(mask.pitch * height),
                                     width,
                                     height,
                                     mask.pitch);

            // Buffer for reading compressed chunk data in
            auto raw_chunk_buffer =
              std::vector<uint8_t>(width * height * sizeof(pixel_t));

            // Let all threads do setup tasks before reading starts
            cpu_sync.arrive_and_wait();
            CudaEvent start, copy, post, postcopy, end;

            // Get the time the lastimage was received to avoid waiting for too long
            auto last_image_received = std::chrono::high_resolution_clock::now();

            while (!stop_token.stop_requested()) {
                auto image_num = next_image.fetch_add(1);
                if (image_num >= num_images) {
                    break;
                }
                auto offset_image_num = image_num + parser.get<uint32_t>("start-index");
                {
                    // TODO:
                    //  - Counting time like this does not work efficiently
                    //    because it might not be the "next" image that
                    //    gets the lock.

                    // Lock so we don't duplicate wait count, and also
                    // because we don't know if the HDF5 function is threadsafe
                    std::scoped_lock lock(reader_mutex);
                    auto swmr_wait_start_time =
                      std::chrono::high_resolution_clock::now();

                    // Check that our image is available and wait if not
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

                        // Sleep for a bit to avoid busy-waiting
                        std::this_thread::sleep_for(100ms);
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
                // Fetch the image data from the reader
                while (true) {
                    {
                        std::scoped_lock lock(reader_mutex);
                        buffer =
                          reader.get_raw_chunk(offset_image_num, raw_chunk_buffer);
                    }
                    // /dev/shm we might not have an atomic write
                    if (buffer.size() == 0) {
                        fmt::print(fmt::runtime(
                          "\033[1mRace Condition?!?? Got buffer size 0 for image "
                          "{image_num}. "
                          "Sleeping.\033[0m\n"));
                        std::this_thread::sleep_for(100ms);
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
                    bshuf_decompress_lz4(
                      buffer.data() + 12, host_image.get(), width * height, 2, 0);
                    break;
                case Reader::ChunkCompression::BYTE_OFFSET_32:
                    // decompress_byte_offset<pixel_t>(buffer,
                    //                                 {host_image.get(), width * height});
                    decompress_byte_offset<pixel_t>(
                      buffer,
                      {host_image.get(),
                       static_cast<std::span<short unsigned int>::size_type>(
                         width * height)});
                    // std::copy(buffer.begin(), buffer.end(), host_image.get());
                    // std::exit(1);
                    break;
                }
                start.record(stream);
                // Copy the image to GPU
                CUDA_CHECK(cudaMemcpy2DAsync(device_image.get(),
                                             device_image.pitch_bytes(),
                                             host_image.get(),
                                             width * sizeof(pixel_t),
                                             width * sizeof(pixel_t),
                                             height,
                                             cudaMemcpyHostToDevice,
                                             stream));
                copy.record(stream);
#pragma endregion Decompression

#pragma region Spotfinding
                // When done, launch the spotfind kernel
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

                // Copy the results buffer back to the CPU
                CUDA_CHECK(cudaMemcpy2DAsync(host_results.get(),
                                             width * sizeof(uint8_t),
                                             device_results.get(),
                                             device_results.pitch_bytes(),
                                             width * sizeof(uint8_t),
                                             height,
                                             cudaMemcpyDeviceToHost,
                                             stream));
                postcopy.record(stream);
                // Now, wait for stream to finish
                CUDA_CHECK(cudaStreamSynchronize(stream));
#pragma endregion Spotfinding

#pragma region Connected Components
                std::unique_ptr<ConnectedComponents> connected_components_2d =
                  std::make_unique<ConnectedComponents>(
                    host_results.get(), host_image.get(), width, height, min_spot_size);

                auto boxes = connected_components_2d->get_boxes();
                size_t num_strong_pixels =
                  connected_components_2d->get_num_strong_pixels();
                size_t num_strong_pixels_filtered =
                  connected_components_2d->get_num_strong_pixels_filtered();

                // If this is a rotation dataset, store the connected component slice
                if (rotation_slices) {
                    // Lock the mutex to protect the map
                    std::lock_guard<std::mutex> lock(rotation_slices_mutex);
                    // Store the connected components slice in the map
                    (*rotation_slices)[offset_image_num] =
                      std::move(connected_components_2d);
                }

                end.record(stream);
                // Now, wait for stream to finish
                CUDA_CHECK(cudaStreamSynchronize(stream));

                if (do_writeout) {
                    // Build an image buffer
                    auto buffer =
                      std::vector<std::array<uint8_t, 3>>(width * height, {0, 0, 0});
                    constexpr std::array<uint8_t, 3> color_pixel{255, 0, 0};

                    for (int y = 0, k = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x, ++k) {
                            uint8_t graysc_value = std::max(
                              0.0f, 255.99f - static_cast<float>(host_image[k]) * 10);
                            buffer[k] = {graysc_value, graysc_value, graysc_value};
                        }
                    }
                    // Go over each shoebox and write a square
                    // for (auto box : boxes) {
                    for (int i = 0; i < boxes.size(); ++i) {
                        auto &box = boxes[i];
                        constexpr std::array<uint8_t, 3> color_shoebox{0, 0, 255};

                        // edgeMin/edgeMax define how thick the border is
                        constexpr int edgeMin = 5, edgeMax = 7;
                        for (int edge = edgeMin; edge <= edgeMax; ++edge) {
                            for (int x = box.l - edge; x <= box.r + edge; ++x) {
                                buffer[width * (box.t - edge) + x] = color_shoebox;
                                buffer[width * (box.b + edge) + x] = color_shoebox;
                            }
                            for (int y = box.t - edge; y <= box.b + edge; ++y) {
                                buffer[width * y + box.l - edge] = color_shoebox;
                                buffer[width * y + box.r + edge] = color_shoebox;
                            }
                        }
                    }
                    // Go over everything again, so that strong spots are visible over the boxes
                    for (int y = 0, k = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x, ++k) {
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
                    // Also write a list of pixels out here
                    auto out =
                      fmt::output_file(fmt::format("pixels_{:05d}.txt", image_num));
                    for (int y = 0, k = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x, ++k) {
                            if (host_results[k]) {
                                out.print("{:4d}, {:4d}\n", x, y);
                            }
                        }
                    }
                }

                // Check if pipeHandler was initialized
                if (pipeHandler != nullptr) {
                    // Create a JSON object to store the data
                    json json_data = {{"num_strong_pixels", num_strong_pixels},
                                      {"file", args.file},
                                      {"file-number", image_num},
                                      {"n_spots_total", boxes.size()}};
                    // Send the JSON data through the pipe
                    pipeHandler->sendData(json_data);
                }
#pragma endregion Connected Components

#pragma region Validation
                if (do_validate) {
                    // Count the number of pixels
                    size_t num_strong_pixels = 0;
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            if (host_results[x + width * y]) {
                                ++num_strong_pixels;
                            }
                        }
                    }
                    auto spotfinder = StandaloneSpotfinder(width, height);
                    // Read the image into a vector
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
                          num_strong_pixels);
                    } else {
                        fmt::print(
                          "Thread {:2d}, Image {:4d}: Compared: "
                          "\033[1;31mMismatch ({} px from kernel)\033[0m\n",
                          thread_id,
                          image_num,
                          num_strong_pixels);
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
#pragma endregion Validation
                // auto image_num = next_image.fetch_add(1);
                completed_images += 1;
            }
        });
    }
    // For now, just wait on all threads to finish
    for (auto &thread : threads) {
        thread.join();
    }

#pragma region 3D Connected Components
    // After all threads have finished processing slices
    if (rotation_slices) {
        logger.info("Processing 3D spots");

        // Step 1: Convert rotation_slices map to a vector
        std::vector<std::unique_ptr<ConnectedComponents>> slices;
        for (auto &[image_num, connected_components] : *rotation_slices) {
            slices.push_back(std::move(connected_components));
        }

        // Step 2: Call find_3d_components
        constexpr uint max_peak_centroid_separation = 2;  // Hardcoded for now
        auto reflections_3d = ConnectedComponents::find_3d_components(
          slices, width, height, min_spot_size_3d, max_peak_centroid_separation);

        // Step 3: Output the 3D reflections
        logger.info(
          fmt::format("Found {} spots", fmt::styled(reflections_3d.size(), fmt_cyan)));

        if (do_writeout) {
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

                // Print all members of the reflection
                // Print x bounds
                out << "X: [" << reflection.get_x_min() << ", "
                    << reflection.get_x_max() << "] ";
                // Print y bounds
                out << "Y: [" << reflection.get_y_min() << ", "
                    << reflection.get_y_max() << "] ";
                // Print z bounds
                out << "Z: [" << reflection.get_z_min() << ", "
                    << reflection.get_z_max() << "] ";
                // Print Centre of Mass
                out << "COM: (" << x << ", " << y << ", " << z << ")\n";
            }
            logger.flush();  // Flush to ensure all messages printed before continuing
        }

        if (save_to_h5) {
            // Step 4: Write the 3D reflections to a `.h5` file using ReflectionTable
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
                // Add the 3D reflections to the table
                table.add_column(
                  "xyzobs.px.value", reflections_3d.size(), 3, flat_coms);
                // Map each reflection to the generated experiment ID
                std::vector<int> id(reflections_3d.size(),
                                    table.get_experiment_ids()[0]);
                table.add_column("id", reflections_3d.size(), 1, id);

                // Write the table to an HDF5 file
                table.write("results_ffs.h5", "dials/processing/group_0");
                logger.info("Succesfully wrote 3D reflections to HDF5 file");
            } catch (const std::exception &e) {
                logger.error("Error writing data to HDF5 file: {}", e.what());
            } catch (...) {
                logger.error("Unknown error writing data to HDF5 file");
            }
        }

        logger.info("3D spot analysis complete");
    }
#pragma endregion 3D Connected Components

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
}
