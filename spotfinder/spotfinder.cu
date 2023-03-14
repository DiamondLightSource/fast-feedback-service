/**
 * Basic Naive Kernel
 * 
 * Does spotfinding in-kernel, without in-depth performance tweaking.
 * 
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <fmt/core.h>

#include <array>
#include <cassert>
#include <chrono>
#include <memory>
#include <utility>

#include "common.hpp"
#include "h5read.h"
#include "standalone.h"

namespace cg = cooperative_groups;

using namespace fmt;

using pixel_t = H5Read::image_type;

/// One-direction width of kernel. Total kernel span is (K_W * 2 + 1)
constexpr int KERNEL_WIDTH = 3;
/// One-direction height of kernel. Total kernel span is (K_H * 2 + 1)
constexpr int KERNEL_HEIGHT = 3;

__global__ void do_spotfinding_naive(pixel_t *image,
                                     size_t image_pitch,
                                     uint8_t *mask,
                                     size_t mask_pitch,
                                     int width,
                                     int height,
                                     int *result_sum,
                                     size_t *result_sumsq,
                                     uint8_t *result_n,
                                     uint8_t *result_strong) {
    image = image + (image_pitch * height * blockIdx.z);
    result_sum = result_sum + (image_pitch * height * blockIdx.z);
    result_sumsq = result_sumsq + (image_pitch * height * blockIdx.z);
    result_n = result_n + (mask_pitch * height * blockIdx.z);
    result_strong = result_strong + (mask_pitch * height * blockIdx.z);

    auto block = cg::this_thread_block();
    // auto warp = cg::tiled_partition<32>(block);
    // int warpId = warp.meta_group_rank();
    // int lane = warp.thread_rank();

    uint sum = 0;
    size_t sumsq = 0;
    uint8_t n = 0;

    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    // Don't calculate for masked pixels
    bool px_is_valid = mask[y * mask_pitch + x] != 0;
    pixel_t this_pixel = image[y * image_pitch + x];

    if (px_is_valid) {
        for (int row = max(0, y - KERNEL_HEIGHT);
             row < min(y + KERNEL_HEIGHT + 1, height);
             ++row) {
            int row_offset = image_pitch * row;
            int mask_offset = mask_pitch * row;
            for (int col = max(0, x - KERNEL_WIDTH);
                 col < min(x + KERNEL_WIDTH + 1, width);
                 ++col) {
                pixel_t pixel = image[row_offset + col];
                uint8_t mask_pixel = mask[mask_offset + col];
                if (mask_pixel) {
                    sum += pixel;
                    sumsq += pixel * pixel;
                    n += 1;
                }
            }
        }
    }

    if (x < width && y < height) {
        result_sum[x + image_pitch * y] = sum;
        result_sumsq[x + image_pitch * y] = sumsq;
        result_n[x + mask_pitch * y] = n;

        // Calculate the thresholding
        if (px_is_valid) {
            constexpr float n_sig_s = 3.0f;
            constexpr float n_sig_b = 6.0f;

            float sum_f = static_cast<float>(sum);
            float sumsq_f = static_cast<float>(sumsq);

            float mean = sum_f / n;
            float variance = (n * sumsq_f - (sum_f * sum_f)) / (n * (n - 1));
            float dispersion = variance / mean;
            float background_threshold = 1 + n_sig_b * sqrt(2.0f / (n - 1));
            bool not_background = dispersion > background_threshold;
            float signal_threshold = mean + n_sig_s * sqrt(mean);
            bool is_signal = this_pixel > signal_threshold;
            bool is_strong_pixel = not_background && is_signal;
            result_strong[x + mask_pitch * y] = is_strong_pixel;
        } else {
            result_strong[x + mask_pitch * y] = 0;
        }
    }
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

    auto args = parser.parse_args(argc, argv);
    uint32_t batch_size = parser.get<uint32_t>("batch");
    if (batch_size < 1) {
        print("Error: Batch size must be >= 1\n");
        std::exit(1);
    }
    bool do_validate = parser.get<bool>("validate");

    auto reader = args.file.empty() ? H5Read() : H5Read(args.file);

    int height = reader.image_shape()[0];
    int width = reader.image_shape()[1];

    // Work out how many blocks this is
    dim3 thread_block_size{32, 16};
    dim3 blocks_dims{
      static_cast<unsigned int>(ceilf((float)width / thread_block_size.x)),
      static_cast<unsigned int>(ceilf((float)height / thread_block_size.y)),
      batch_size};
    const int num_threads_per_block = thread_block_size.x * thread_block_size.y;
    const int num_blocks = blocks_dims.x * blocks_dims.y * blocks_dims.z;
    print("Image:   {:4d} x {:4d} x {:2d} = {} px\n",
          width,
          height,
          batch_size,
          width * height * batch_size);
    print("Threads: {:4d} x {:<4d} = {}\n",
          thread_block_size.x,
          thread_block_size.y,
          num_threads_per_block);
    print("Blocks:  {:4d} x {:<4d} x {:2d} = {}\n",
          blocks_dims.x,
          blocks_dims.y,
          blocks_dims.z,
          num_blocks);

    // Create a host memory area to read the image into
    // auto host_image = std::make_unique<pixel_t[]>(width * height);
    auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height * batch_size);

    // Device-side pitched storage for image data
    auto [dev_image, device_pitch] =
      make_cuda_pitched_malloc<pixel_t>(width, height * batch_size);
    auto [dev_mask, device_mask_pitch] =
      make_cuda_pitched_malloc<uint8_t>(width, height * batch_size);
    print("Allocated device memory. Pitch = {} vs naive {}\n", device_pitch, width);

    // Managed memory areas for results
    auto result_sum = make_cuda_managed_malloc<int>(device_pitch * height * batch_size);
    auto result_sumsq =
      make_cuda_managed_malloc<size_t>(device_pitch * height * batch_size);
    auto result_n =
      make_cuda_managed_malloc<uint8_t>(device_mask_pitch * height * batch_size);
    // auto result_strong =
    //   make_cuda_managed_malloc<uint8_t>(device_mask_pitch * height * batch_size);
    auto dev_result_strong =
      make_cuda_malloc<uint8_t>(device_mask_pitch * height * batch_size);
    auto result_strong =
      make_cuda_pinned_malloc<uint8_t>(device_mask_pitch * height * batch_size);
    // Make sure to clear these completely
    cudaMemset(result_sum.get(), 0, sizeof(int) * device_pitch * height * batch_size);
    cudaMemset(
      result_sumsq.get(), 0, sizeof(size_t) * device_pitch * height * batch_size);
    cudaMemset(
      result_n.get(), 0, sizeof(uint8_t) * device_mask_pitch * height * batch_size);
    cudaMemset(result_strong.get(),
               0,
               sizeof(uint8_t) * device_mask_pitch * height * batch_size);
    cudaMemset(dev_result_strong.get(),
               0,
               sizeof(uint8_t) * device_mask_pitch * height * batch_size);
    cudaDeviceSynchronize();
    cuda_throw_error();

    CudaEvent pre_load, start, memcpy, kernel, all;

    size_t mask_sum = 0;
    if (reader.get_mask()) {
        mask_sum = 0;
        for (size_t i = 0; i < width * height; ++i) {
            if (reader.get_mask().value()[i]) {
                mask_sum += 1;
            }
        }
        start.record();
        cudaMemcpy2D(dev_mask.get(),
                     device_mask_pitch,
                     reader.get_mask()->data(),
                     width,
                     width,
                     height,
                     cudaMemcpyHostToDevice);
        cuda_throw_error();
    } else {
        mask_sum = width * height;
        start.record();
        cudaMemset(dev_mask.get(), 1, device_mask_pitch * height);
        cuda_throw_error();
    }
    memcpy.record();
    memcpy.synchronize();

    float memcpy_time = memcpy.elapsed_time(start);
    print("Uploaded mask ({:.2f} Mpx) in {:.2f} ms ({:.1f} GBps)\n",
          static_cast<float>(mask_sum) / 1e6,
          memcpy_time,
          GBps(memcpy_time, width * height));

    print("\nProcessing {} Images\n\n", reader.get_number_of_images());
    auto spotfinder = StandaloneSpotfinder(width, height);

    auto all_images_start_time = std::chrono::high_resolution_clock::now();
    for (size_t image_id = 0; image_id < reader.get_number_of_images();
         image_id += batch_size) {
        if (args.image_number.has_value()
            && (args.image_number.value() < image_id
                || args.image_number.value() >= image_id + batch_size)) {
            continue;
        }

        // How many images to do in this batch. May be less than batch size.
        size_t num_images =
          min(image_id + batch_size, reader.get_number_of_images()) - image_id;
        blocks_dims.z = num_images;
        size_t pixels_processed = width * height * num_images;

        if (batch_size > 1) {
            print("Images [{}-{}]:\n", image_id, image_id + num_images - 1);
        } else {
            print("Image  {}:\n", image_id);
        }

        pre_load.record();
        pre_load.synchronize();
        // print("Num images: {}\n", num_images);
        for (size_t offset_id = 0; offset_id < num_images; ++offset_id) {
            reader.get_image_into(image_id + offset_id,
                                  host_image.get() + (width * height * offset_id));
        }

        // Copy the image(s) to GPU
        start.record();
        cudaMemcpy2DAsync(dev_image.get(),
                          device_pitch * sizeof(pixel_t),
                          host_image.get(),
                          width * sizeof(pixel_t),
                          width * sizeof(pixel_t),
                          height * num_images,
                          cudaMemcpyHostToDevice);
        memcpy.record();
        cuda_throw_error();

        do_spotfinding_naive<<<blocks_dims, thread_block_size>>>(dev_image.get(),
                                                                 device_pitch,
                                                                 dev_mask.get(),
                                                                 device_mask_pitch,
                                                                 width,
                                                                 height,
                                                                 result_sum.get(),
                                                                 result_sumsq.get(),
                                                                 result_n.get(),
                                                                 result_strong.get());
        kernel.record();
        all.record();
        cuda_throw_error();
        cudaDeviceSynchronize();

        print("    Read Time: \033[1m{:6.2f}\033[0m ms \033[37m{:>11}\033[0m\n",
              start.elapsed_time(pre_load),
              format("({:4.1f} GBps)",
                     GBps<pixel_t>(start.elapsed_time(pre_load), pixels_processed)));
        print("  Upload Time: \033[1m{:6.2f}\033[0m ms \033[37m({:4.1f} GBps)\033[0m\n",
              memcpy.elapsed_time(start),
              GBps<pixel_t>(memcpy.elapsed_time(start), pixels_processed));
        print("  Kernel Time: \033[1m{:6.2f}\033[0m ms \033[37m{:>11}\033[0m\n",
              kernel.elapsed_time(memcpy),
              format("({:.1f} GBps)",
                     GBps<pixel_t>(kernel.elapsed_time(memcpy), pixels_processed)));
        print("               ════════\n");
        print("        Total: \033[1m{:6.2f}\033[0m ms {:>11}\n",
              all.elapsed_time(pre_load),
              format("({:.1f} GBps)",
                     GBps<pixel_t>(all.elapsed_time(pre_load), pixels_processed)));

        auto strong = count_nonzero(result_strong.get(),
                                    width,
                                    static_cast<int>(height * num_images),
                                    device_mask_pitch);
        print("       Strong: {} px\n", strong);

        if (do_validate) {
            auto start_time = std::chrono::high_resolution_clock::now();
            size_t mismatch_x = 0, mismatch_y = 0;

            for (size_t offset_id = 0; offset_id < num_images; ++offset_id) {
                size_t offset_plain = width * height * offset_id;
                size_t offset_image = device_pitch * height * offset_id;
                size_t offset_mask = device_mask_pitch * height * offset_id;

                if (batch_size > 1) {
                    print("  Image {}:\n", image_id + offset_id);
                }
                // Read the image into a vector
                auto converted_image = std::vector<double>{
                  host_image.get() + width * height * offset_id,
                  host_image.get() + width * height * (offset_id + 1)};
                auto dials_strong = spotfinder.standard_dispersion(
                  converted_image, reader.get_mask().value_or(span<uint8_t>{}));
                auto end_time = std::chrono::high_resolution_clock::now();
                size_t dials_results =
                  count_nonzero(dials_strong, width, height, width);
                float validation_time_ms =
                  std::chrono::duration_cast<std::chrono::duration<double>>(
                    end_time - start_time)
                    .count()
                  * 1000;
                print("        Dials: {} px in {:.0f} ms CPU time\n",
                      dials_results,
                      validation_time_ms);
                bool validation_matches =
                  compare_results(dials_strong.data(),
                                  width,
                                  result_strong.get() + offset_mask,
                                  device_mask_pitch,
                                  width,
                                  height,
                                  &mismatch_x,
                                  &mismatch_y);

                if (validation_matches) {
                    print("     Compared: \033[32mMatch\033[0m\n");
                } else {
                    print("     Compared: \033[1;31mMismatch\033[0m\n");

                    mismatch_x = max(static_cast<int>(mismatch_x) - 8, 0);
                    mismatch_y = max(static_cast<int>(mismatch_y) - 8, 0);

                    print("Data:\n");
                    draw_image_data(host_image.get() + offset_plain,
                                    mismatch_x,
                                    mismatch_y,
                                    16,
                                    16,
                                    width,
                                    height);
                    print("Strong From DIALS:\n");
                    draw_image_data(dials_strong.data() + offset_plain,
                                    mismatch_x,
                                    mismatch_y,
                                    16,
                                    16,
                                    width,
                                    height);
                    print("Strong From kernel:\n");
                    draw_image_data(result_strong.get() + offset_mask,
                                    mismatch_x,
                                    mismatch_y,
                                    16,
                                    16,
                                    device_mask_pitch,
                                    height);
                    // print("Resultant N:\n");
                    print("Sum From kernel:\n");
                    draw_image_data(result_sum.get() + offset_image,
                                    mismatch_x,
                                    mismatch_y,
                                    16,
                                    16,
                                    device_pitch,
                                    height);
                    print("Sum² From kernel:\n");
                    draw_image_data(result_sumsq.get() + offset_image,
                                    mismatch_x,
                                    mismatch_y,
                                    16,
                                    16,
                                    device_pitch,
                                    height);
                    print("Mask:\n");
                    draw_image_data(reader.get_mask().value().data(),
                                    mismatch_x,
                                    mismatch_y,
                                    16,
                                    16,
                                    width,
                                    height);
                }
            }
        }
        print("\n\n");
    }
    float total_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - all_images_start_time)
        .count();
    print(
      "{} images in {:.2f} s ({:.2f} GBps)\n",
      reader.get_number_of_images(),
      total_time,
      GBps<pixel_t>(total_time * 1000, width * height * reader.get_number_of_images()));
}
