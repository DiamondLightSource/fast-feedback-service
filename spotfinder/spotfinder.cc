#include "spotfinder.h"

#include <bitshuffle.h>
#include <fmt/core.h>
#include <lodepng.h>
#include <nppi_filtering_functions.h>

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
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <ranges>
#include <stop_token>
#include <thread>
#include <utility>

#include "cbfread.hpp"
#include "common.hpp"
#include "h5read.h"
#include "shmread.hpp"
#include "standalone.h"

using namespace fmt;
using namespace std::chrono_literals;

// Global stop token for picking up user cancellation
std::stop_source global_stop;

// Function for passing to std::signal to register the stop request
extern "C" void stop_processing(int sig) {
    if (global_stop.stop_requested()) {
        // We already requested before, but we want it faster. Abort.
        std::quick_exit(1);
    } else {
        print("Running interrupted by user request\n");
        global_stop.request_stop();
    }
}

/// Very basic comparison operator for convenience
auto operator==(const int2 &left, const int2 &right) -> bool {
    return left.x == right.x && left.y == right.y;
}

struct Reflection {
    int l, t, r, b;
    int num_pixels = 0;
};

template <typename T>
struct PitchedMalloc {
  public:
    using value_type = T;
    PitchedMalloc(std::shared_ptr<T[]> data, size_t width, size_t height, size_t pitch)
        : _data(data), width(width), height(height), pitch(pitch) {}

    PitchedMalloc(size_t width, size_t height) : width(width), height(height) {
        auto [alloc, alloc_pitch] = make_cuda_pitched_malloc<T>(width, height);
        _data = alloc;
        pitch = alloc_pitch;
    }

    auto get() {
        return _data.get();
    }
    auto size_bytes() -> size_t const {
        return pitch * height * sizeof(T);
    }
    auto pitch_bytes() -> size_t const {
        return pitch * sizeof(T);
    }

    std::shared_ptr<T[]> _data;
    size_t width;
    size_t height;
    size_t pitch;
};

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
    print("Uploaded mask ({:.2f} Mpx) in {:.2f} ms ({:.1f} GBps)\n",
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

/// Handle setting up an NppStreamContext from a specific stream
auto create_npp_context_from_stream(const CudaStream &stream) -> NppStreamContext {
    NppStreamContext npp_context;
    npp_context.hStream = stream;
    CUDA_CHECK(cudaGetDevice(&npp_context.nCudaDeviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&npp_context.nCudaDevAttrComputeCapabilityMajor,
                                      cudaDevAttrComputeCapabilityMajor,
                                      npp_context.nCudaDeviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&npp_context.nCudaDevAttrComputeCapabilityMinor,
                                      cudaDevAttrComputeCapabilityMinor,
                                      npp_context.nCudaDeviceId));
    CUDA_CHECK(cudaStreamGetFlags(npp_context.hStream, &npp_context.nStreamFlags));
    cudaDeviceProp oDeviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&oDeviceProperties, npp_context.nCudaDeviceId));

    npp_context.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
    npp_context.nMaxThreadsPerMultiProcessor =
      oDeviceProperties.maxThreadsPerMultiProcessor;
    npp_context.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
    npp_context.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;
    return npp_context;
}

void wait_for_ready_for_read(const std::string &path,
                             std::function<bool(const std::string &)> checker,
                             float timeout = 120.0f) {
    if (!checker(path)) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto message_prefix =
          format("Waiting for \033[1;35m{}\033[0m to be ready for read", path);
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
            print("\r{}  {} [{:4.1f} s] ", message_prefix, ball[i], wait_time);
            i = (i + 1) % ball.size();
            std::cout << std::flush;

            if (wait_time > timeout) {
                print("\nError: Waited too long for read availability\n");
                std::exit(1);
            }
            std::this_thread::sleep_for(80ms);
        }
        print("\n");
    }
}

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
        print("PipeHandler initialized with pipe_fd: {}\n", pipe_fd);
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
    void sendData(const nlohmann::json &json_data) {
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
            // print("Data sent through the pipe: {}\n", stringified_json);
        }
    }
};

int main(int argc, char **argv) {
    // Parse arguments and get our H5Reader
    auto parser = CUDAArgumentParser();
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
      .help("Reflections with a pixel count below this will be discarded.")
      .metavar("N")
      .default_value<uint32_t>(2)
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

    auto args = parser.parse_args(argc, argv);
    bool do_validate = parser.get<bool>("validate");
    bool do_writeout = parser.get<bool>("writeout");
    int pipe_fd = parser.get<int>("pipe_fd");
    float wait_timeout = parser.get<float>("timeout");

    uint32_t num_cpu_threads = parser.get<uint32_t>("threads");
    if (num_cpu_threads < 1) {
        print("Error: Thread count must be >= 1\n");
        std::exit(1);
    }
    uint32_t min_spot_size = parser.get<uint32_t>("min-spot-size");

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
            print("Error: CBF reading must specify --images\n");
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

    int height = reader.image_shape()[0];
    int width = reader.image_shape()[1];

    std::signal(SIGINT, stop_processing);

    // Work out how many blocks this is
    dim3 gpu_thread_block_size{32, 16};
    dim3 blocks_dims{
      static_cast<unsigned int>(ceilf((float)width / gpu_thread_block_size.x)),
      static_cast<unsigned int>(ceilf((float)height / gpu_thread_block_size.y))};
    const int num_threads_per_block = gpu_thread_block_size.x * gpu_thread_block_size.y;
    const int num_blocks = blocks_dims.x * blocks_dims.y * blocks_dims.z;
    print("Image:       {:4d} x {:4d} = {} px\n", width, height, width * height);
    print("GPU Threads: {:4d} x {:<4d} = {}\n",
          gpu_thread_block_size.x,
          gpu_thread_block_size.y,
          num_threads_per_block);
    print("Blocks:      {:4d} x {:<4d} x {:2d} = {}\n",
          blocks_dims.x,
          blocks_dims.y,
          blocks_dims.z,
          num_blocks);
    print("Running with {} CPU threads\n", num_cpu_threads);

    auto mask = upload_mask(reader);

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

            // Initialise NPP buffers
            int npp_buffer_size = 0;
            NPP_CHECK(nppiLabelMarkersUFGetBufferSize_32u_C1R({width, height},
                                                              &npp_buffer_size));
            auto device_label_buffer = make_cuda_malloc<Npp8u>(npp_buffer_size);
            auto device_label_dest = PitchedMalloc<Npp32u>(width, height);
            auto npp_context = create_npp_context_from_stream(stream);

            // Buffer for reading compressed chunk data in
            auto raw_chunk_buffer =
              std::vector<uint8_t>(width * height * sizeof(pixel_t));

            // Allocate buffers for DIALS-style extraction
            auto px_coords = std::vector<int2>();
            auto px_values = std::vector<pixel_t>();
            auto px_kvals = std::vector<size_t>();

            // Let all threads do setup tasks before reading starts
            cpu_sync.arrive_and_wait();
            CudaEvent start, copy, post, postcopy, end;

            while (!stop_token.stop_requested()) {
                auto image_num = next_image.fetch_add(1);
                if (image_num >= num_images) {
                    break;
                }
                {
                    // TODO:
                    //  - This loop does not handle the stop token
                    //  - Counting time like this does not work efficiently
                    //    because it might not be the "next" image that
                    //    gets the lock.
                    //  - This freezes if the collection is stopped;
                    //    continue implementing timout feature
                    //
                    // Lock so we don't duplicate wait count, and also
                    // because we don't know if the HDF5 function is threadsafe
                    std::scoped_lock lock(reader_mutex);
                    auto swmr_wait_start_time =
                      std::chrono::high_resolution_clock::now();
                    // Check that our image is available and wait if not
                    while (!reader.is_image_available(image_num)) {
                        std::this_thread::sleep_for(100ms);
                    }
                    time_waiting_for_images +=
                      std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now()
                        - swmr_wait_start_time)
                        .count();
                }
                // Sized buffer for the actual data read from file
                span<uint8_t> buffer;
                // Fetch the image data from the reader
                while (true) {
                    {
                        std::scoped_lock lock(reader_mutex);
                        buffer = reader.get_raw_chunk(image_num, raw_chunk_buffer);
                    }
                    // /dev/shm we might not have an atomic write
                    if (buffer.size() == 0) {
                        print(
                          "\033[1mRace Condition?!?? Got buffer size 0 for image "
                          "{image_num}. "
                          "Sleeping.\033[0m\n");
                        std::this_thread::sleep_for(100ms);
                        continue;
                    }
                    break;
                }
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
                    decompress_byte_offset<pixel_t>(buffer,
                                                    {host_image.get(), width * height});
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
                // When done, launch the spotfind kernel
                // do_spotfinding_naive<<<blocks_dims, gpu_thread_block_size, 0, stream>>>(
                call_do_spotfinding_naive(blocks_dims,
                                          gpu_thread_block_size,
                                          0,
                                          stream,
                                          device_image.get(),
                                          device_image.pitch,
                                          mask.get(),
                                          mask.pitch,
                                          width,
                                          height,
                                          device_results.get());
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

                // Manually reproduce what the DIALS connected components does
                // Start with the behaviour of the PixelList class:
                size_t num_strong_pixels = 0;
                px_values.clear();
                px_coords.clear();
                px_kvals.clear();

                for (int y = 0, k = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x, ++k) {
                        if (host_results[k]) {
                            px_coords.emplace_back(x, y);
                            px_values.push_back(host_image[k]);
                            px_kvals.push_back(k);
                            ++num_strong_pixels;
                        }
                    }
                }

                auto graph =
                  boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>{
                    px_values.size()};

                // Index for next pixel to search when looking for pixels
                // below the current one. This will only ever increase, because
                // we are guaranteed to always look for one after the last found
                // pixel.
                int idx_pixel_below = 1;

                for (int i = 0; i < static_cast<int>(px_coords.size()) - 1; ++i) {
                    auto coord = px_coords[i];
                    auto coord_right = int2{coord.x + 1, coord.y};
                    auto k = px_kvals[i];

                    if (px_coords[i + 1] == coord_right) {
                        // Since we generate strong pixels coordinates horizontally,
                        // if there is a pixel to the right then it is guaranteed
                        // to be the next one in the list. Connect these.
                        boost::add_edge(i, i + 1, graph);
                    }
                    // Now, check the pixel directly below this one. We need to scan
                    // to find it, because _if_ there is a matching strong pixel,
                    // then we don't know how far ahead it is in the coordinates array
                    if (coord.y < height - 1) {
                        auto coord_below = int2{coord.x, coord.y + 1};
                        auto k_below = k + width;
                        // int idx = i + 1;
                        while (idx_pixel_below < px_coords.size() - 1
                               && px_kvals[idx_pixel_below] < k_below) {
                            ++idx_pixel_below;
                        }
                        // Either we've got the pixel below, past that - or the
                        // last pixel in the coordinate set.
                        if (px_coords[idx_pixel_below] == coord_below) {
                            boost::add_edge(i, idx_pixel_below, graph);
                        }
                    }
                }
                auto labels = std::vector<int>(boost::num_vertices(graph));
                auto num_labels = boost::connected_components(graph, labels.data());

                auto boxes = std::vector<Reflection>(num_labels, {width, height, 0, 0});

                assert(labels.size() == px_coords.size());
                for (int i = 0; i < labels.size(); ++i) {
                    auto label = labels[i];
                    auto coord = px_coords[i];
                    Reflection &box = boxes[label];
                    box.l = std::min(box.l, coord.x);
                    box.r = std::max(box.r, coord.x);
                    box.t = std::min(box.t, coord.y);
                    box.b = std::max(box.b, coord.y);
                    box.num_pixels += 1;
                }

                // Filter shoeboxes
                if (min_spot_size > 0) {
                    std::vector<Reflection> filtered_boxes;
                    for (auto &box : boxes) {
                        if (box.num_pixels >= min_spot_size) {
                            filtered_boxes.emplace_back(box);
                        }
                    }
                    boxes = std::move(filtered_boxes);
                }
                // // Do the connected component calculations
                // NPP_CHECK(nppiLabelMarkersUF_8u32u_C1R_Ctx(device_results.get(),
                //                                            device_results.pitch,
                //                                            device_label_dest.get(),
                //                                            device_label_dest.pitch,
                //                                            {width, height},
                //                                            nppiNormL1,
                //                                            device_label_buffer.get(),
                //                                            npp_context));
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
                            if (host_results[k]) {
                                buffer[k] = color_pixel;
                            }
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
                    lodepng::encode(format("image_{:05d}.png", image_num),
                                    reinterpret_cast<uint8_t *>(buffer.data()),
                                    width,
                                    height,
                                    LCT_RGB);
                }

                // Check if pipeHandler was initialized
                if (pipeHandler != nullptr) {
                    // Create a JSON object to store the data
                    nlohmann::json json_data = {
                      {"num_strong_pixels", num_strong_pixels},
                      {"file", args.file},
                      {"file-number", image_num},
                      {"n_spots_total", boxes.size()}};
                    // Send the JSON data through the pipe
                    pipeHandler->sendData(json_data);
                }

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
                      converted_image, reader.get_mask().value_or(span<uint8_t>{}));
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
                        print(
                          "Thread {:2d}, Image {:4d}: Compared: \033[32mMatch {} "
                          "px\033[0m\n",
                          thread_id,
                          image_num,
                          num_strong_pixels);
                    } else {
                        print(
                          "Thread {:2d}, Image {:4d}: Compared: "
                          "\033[1;31mMismatch ({} px from kernel)\033[0m\n",
                          thread_id,
                          image_num,
                          num_strong_pixels);
                    }

                } else {
                    if (num_cpu_threads == 1) {
                        print(
                          "Thread {:2d} finished image {:4d}\n"
                          "       Copy: {:5.1f} ms\n"
                          "     Kernel: {:5.1f} ms\n"
                          "  Post Copy: {:5.1f} ms\n"
                          "       Post: {:5.1f} ms\n"
                          "             ════════\n"
                          "     Total:  {:5.1f} ms ({:.1f} GBps)\n"
                          "    {} strong pixels in {} reflections\n",
                          thread_id,
                          image_num,
                          copy.elapsed_time(start),
                          post.elapsed_time(start),
                          postcopy.elapsed_time(post),
                          end.elapsed_time(postcopy),
                          end.elapsed_time(start),
                          GBps<pixel_t>(end.elapsed_time(start), width * height),
                          bold(num_strong_pixels),
                          bold(boxes.size()));
                    } else {
                        print(
                          "Thread {:2d} finished image {:4d} with {} pixels in {} "
                          "reflections\n",
                          thread_id,
                          image_num,
                          num_strong_pixels,
                          boxes.size());
                    }
                }
                // auto image_num = next_image.fetch_add(1);
                completed_images += 1;
            }
        });
    }
    // For now, just wait on all threads to finish
    for (auto &thread : threads) {
        thread.join();
    }

    float total_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - all_images_start_time)
        .count();
    print(
      "\n{} images in {:.2f} s (\033[1;34m{:.2f} GBps\033[0m) "
      "(\033[1;34m{:.1f} fps\033[0m)\n",
      completed_images,
      total_time,
      GBps<pixel_t>(
        total_time * 1000,
        static_cast<size_t>(width) * static_cast<size_t>(height) * completed_images),
      completed_images / total_time,
      width,
      height);
    if (time_waiting_for_images < 10) {
        print("Total time waiting for images to appear: {:.0f} ms\n",
              time_waiting_for_images * 1000);
    } else {
        print("Total time waiting for images to appear: {:.2f} s\n",
              time_waiting_for_images);
    }
}
