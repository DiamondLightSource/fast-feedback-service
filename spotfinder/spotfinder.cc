/**
 * Basic Naive Kernel
 * 
 * Does spotfinding in-kernel, without in-depth performance tweaking.
 * 
 */

#include "spotfinder.h"

#include <bitshuffle.h>
#include <fmt/core.h>
#include <nppi_filtering_functions.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <memory>
#include <stop_token>
#include <thread>
#include <utility>

#include "common.hpp"
#include "h5read.h"
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
auto upload_mask(H5Read &reader) -> PitchedMalloc<uint8_t> {
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
    auto args = parser.parse_args(argc, argv);
    bool do_validate = parser.get<bool>("validate");
    uint32_t num_cpu_threads = parser.get<uint32_t>("threads");
    if (num_cpu_threads < 1) {
        print("Error: Thread count must be >= 1\n");
        std::exit(1);
    }

    auto reader = args.file.empty() ? H5Read() : H5Read(args.file);
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

    // Spawn the reader threads
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_cpu_threads; ++i) {
        threads.emplace_back([&, i]() {
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

            // Let all threads do setup tasks before reading starts
            cpu_sync.arrive_and_wait();
            CudaEvent start, copy, post, end;

            while (!stop_token.stop_requested()) {
                auto image_num = next_image.fetch_add(1);
                if (image_num >= num_images) {
                    break;
                }
                // Sized buffer for the actual data read from file
                span<uint8_t> buffer;
                // Fetch the image data from the reader
                {
                    std::scoped_lock lock(reader_mutex);
                    buffer = reader.get_raw_chunk(image_num, raw_chunk_buffer);
                }
                // Decompress this data, outside of the mutex
                bshuf_decompress_lz4(
                  buffer.data() + 12, host_image.get(), width * height, 2, 0);
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

                // Do the connected component calculations
                NPP_CHECK(nppiLabelMarkersUF_8u32u_C1R_Ctx(device_results.get(),
                                                           device_results.pitch,
                                                           device_label_dest.get(),
                                                           device_label_dest.pitch,
                                                           {width, height},
                                                           nppiNormL1,
                                                           device_label_buffer.get(),
                                                           npp_context));
                end.record(stream);

                if (do_validate) {
                    // Now, copy the results buffer back to the CPU
                    CUDA_CHECK(cudaMemcpy2DAsync(host_results.get(),
                                                 width * sizeof(uint8_t),
                                                 device_results.get(),
                                                 device_results.pitch_bytes(),
                                                 width * sizeof(uint8_t),
                                                 height,
                                                 cudaMemcpyDeviceToHost,
                                                 stream));
                }
                // Now, wait for stream to finish
                CUDA_CHECK(cudaStreamSynchronize(stream));

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
                          i,
                          image_num,
                          num_strong_pixels);
                    } else {
                        print(
                          "Thread {:2d}, Image {:4d}: Compared: "
                          "\033[1;31mMismatch ({} px from kernel)\033[0m\n",
                          i,
                          image_num,
                          num_strong_pixels);
                    }

                } else {
                    if (num_cpu_threads == 1) {
                        print(
                          "Thread {:2d} finished image {:4d}\n"
                          "       Copy: {:5.1f} ms\n"
                          "     Kernel: {:5.1f} ms\n"
                          "       Post: {:5.1f} ms\n"
                          "             ════════\n"
                          "     Total:  {:5.1f} ms ({:.1f} GBps)\n",
                          i,
                          image_num,
                          copy.elapsed_time(start),
                          post.elapsed_time(start),
                          end.elapsed_time(post),
                          end.elapsed_time(start),
                          GBps<pixel_t>(end.elapsed_time(start), width * height));
                    } else {
                        print("Thread {:2d} finished image {:4d}\n", i, image_num);
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

    // CudaEvent pre_load, start, memcpy, kernel, all;

    // print("\nProcessing {} Images\n\n", reader.get_number_of_images());
    // auto spotfinder = StandaloneSpotfinder(width, height);

    // for (size_t image_id = 0; image_id < reader.get_number_of_images();
    //      image_id += batch_size) {
    //     if (args.image_number.has_value()
    //         && (args.image_number.value() < image_id
    //             || args.image_number.value() >= image_id + batch_size)) {
    //         continue;
    //     }

    //     // How many images to do in this batch. May be less than batch size.
    //     size_t num_images =
    //       min(image_id + batch_size, reader.get_number_of_images()) - image_id;
    //     blocks_dims.z = num_images;
    //     size_t pixels_processed = width * height * num_images;

    //     if (batch_size > 1) {
    //         print("Images [{}-{}]:\n", image_id, image_id + num_images - 1);
    //     } else {
    //         print("Image  {}:\n", image_id);
    //     }

    //     pre_load.record();
    //     pre_load.synchronize();
    //     // print("Num images: {}\n", num_images);
    //     for (size_t offset_id = 0; offset_id < num_images; ++offset_id) {
    //         reader.get_image_into(image_id + offset_id,
    //                               host_image.get() + (width * height * offset_id));
    //     }

    //     // Copy the image(s) to GPU
    //     start.record();
    //     cudaMemcpy2DAsync(dev_image.get(),
    //                       device_pitch * sizeof(pixel_t),
    //                       host_image.get(),
    //                       width * sizeof(pixel_t),
    //                       width * sizeof(pixel_t),
    //                       height * num_images,
    //                       cudaMemcpyHostToDevice);
    //     memcpy.record();
    //     cuda_throw_error();

    //     do_spotfinding_naive<<<blocks_dims, gpu_thread_block_size>>>(
    //       dev_image.get(),
    //       device_pitch,
    //       dev_mask.get(),
    //       device_mask_pitch,
    //       width,
    //       height,
    //       result_sum.get(),
    //       result_sumsq.get(),
    //       result_n.get(),
    //       result_strong.get());
    //     kernel.record();
    //     all.record();
    //     cuda_throw_error();
    //     cudaDeviceSynchronize();

    //     print("    Read Time: \033[1m{:6.2f}\033[0m ms \033[37m{:>11}\033[0m\n",
    //           start.elapsed_time(pre_load),
    //           format("({:4.1f} GBps)",
    //                  GBps<pixel_t>(start.elapsed_time(pre_load), pixels_processed)));
    //     print("  Upload Time: \033[1m{:6.2f}\033[0m ms \033[37m({:4.1f} GBps)\033[0m\n",
    //           memcpy.elapsed_time(start),
    //           GBps<pixel_t>(memcpy.elapsed_time(start), pixels_processed));
    //     print("  Kernel Time: \033[1m{:6.2f}\033[0m ms \033[37m{:>11}\033[0m\n",
    //           kernel.elapsed_time(memcpy),
    //           format("({:.1f} GBps)",
    //                  GBps<pixel_t>(kernel.elapsed_time(memcpy), pixels_processed)));
    //     print("               ════════\n");
    //     print("        Total: \033[1m{:6.2f}\033[0m ms {:>11}\n",
    //           all.elapsed_time(pre_load),
    //           format("({:.1f} GBps)",
    //                  GBps<pixel_t>(all.elapsed_time(pre_load), pixels_processed)));

    //     auto strong = count_nonzero(result_strong.get(),
    //                                 width,
    //                                 static_cast<int>(height * num_images),
    //                                 device_mask_pitch);
    //     print("       Strong: {} px\n", strong);

    //     if (do_validate) {
    //         auto start_time = std::chrono::high_resolution_clock::now();
    //         size_t mismatch_x = 0, mismatch_y = 0;

    //         for (size_t offset_id = 0; offset_id < num_images; ++offset_id) {
    //             size_t offset_plain = width * height * offset_id;
    //             size_t offset_image = device_pitch * height * offset_id;
    //             size_t offset_mask = device_mask_pitch * height * offset_id;

    //             if (batch_size > 1) {
    //                 print("  Image {}:\n", image_id + offset_id);
    //             }
    //             // Read the image into a vector
    //             auto converted_image = std::vector<double>{
    //               host_image.get() + width * height * offset_id,
    //               host_image.get() + width * height * (offset_id + 1)};
    //             auto dials_strong = spotfinder.standard_dispersion(
    //               converted_image, reader.get_mask().value_or(span<uint8_t>{}));
    //             auto end_time = std::chrono::high_resolution_clock::now();
    //             size_t dials_results =
    //               count_nonzero(dials_strong, width, height, width);
    //             float validation_time_ms =
    //               std::chrono::duration_cast<std::chrono::duration<double>>(
    //                 end_time - start_time)
    //                 .count()
    //               * 1000;
    //             print("        Dials: {} px in {:.0f} ms CPU time\n",
    //                   dials_results,
    //                   validation_time_ms);
    //             bool validation_matches =
    //               compare_results(dials_strong.data(),
    //                               width,
    //                               result_strong.get() + offset_mask,
    //                               device_mask_pitch,
    //                               width,
    //                               height,
    //                               &mismatch_x,
    //                               &mismatch_y);

    //             if (validation_matches) {
    //                 print("     Compared: \033[32mMatch\033[0m\n");
    //             } else {
    //                 print("     Compared: \033[1;31mMismatch\033[0m\n");

    //                 mismatch_x = max(static_cast<int>(mismatch_x) - 8, 0);
    //                 mismatch_y = max(static_cast<int>(mismatch_y) - 8, 0);

    //                 print("Data:\n");
    //                 draw_image_data(host_image.get() + offset_plain,
    //                                 mismatch_x,
    //                                 mismatch_y,
    //                                 16,
    //                                 16,
    //                                 width,
    //                                 height);
    //                 print("Strong From DIALS:\n");
    //                 draw_image_data(dials_strong.data() + offset_plain,
    //                                 mismatch_x,
    //                                 mismatch_y,
    //                                 16,
    //                                 16,
    //                                 width,
    //                                 height);
    //                 print("Strong From kernel:\n");
    //                 draw_image_data(result_strong.get() + offset_mask,
    //                                 mismatch_x,
    //                                 mismatch_y,
    //                                 16,
    //                                 16,
    //                                 device_mask_pitch,
    //                                 height);
    //                 // print("Resultant N:\n");
    //                 print("Sum From kernel:\n");
    //                 draw_image_data(result_sum.get() + offset_image,
    //                                 mismatch_x,
    //                                 mismatch_y,
    //                                 16,
    //                                 16,
    //                                 device_pitch,
    //                                 height);
    //                 print("Sum² From kernel:\n");
    //                 draw_image_data(result_sumsq.get() + offset_image,
    //                                 mismatch_x,
    //                                 mismatch_y,
    //                                 16,
    //                                 16,
    //                                 device_pitch,
    //                                 height);
    //                 print("Mask:\n");
    //                 draw_image_data(reader.get_mask().value().data(),
    //                                 mismatch_x,
    //                                 mismatch_y,
    //                                 16,
    //                                 16,
    //                                 width,
    //                                 height);
    //             }
    //         }
    //     }
    //     print("\n\n");
    // }
    float total_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - all_images_start_time)
        .count();
    print(
      "\n{} images in {:.2f} s (\033[1;34m{:.2f} GBps\033[0m) "
      "(\033[1;34m{:.1f} fps\033[0m)\n",
      completed_images,
      total_time,
      GBps<pixel_t>(total_time * 1000, width * height * completed_images),
      completed_images / total_time);
}
