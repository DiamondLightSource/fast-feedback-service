/**
 * Basic Naive Kernel
 * 
 * Does spotfinding in-kernel, without in-depth performance tweaking.
 * 
 */

// #include <bitshuffle.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <lodepng.h>

#include "kernels/erosion.cuh"
#include "kernels/thresholding.cuh"
#include "spotfinder.cuh"

namespace cg = cooperative_groups;

#pragma region Res Mask Functions
/**
 * @brief Function to calculate the distance of a pixel from the beam center.
 * @param x The x-coordinate of the pixel in the image
 * @param y The y-coordinate of the pixel in the image
 * @param center_x The x-coordinate of the pixel beam center in the image
 * @param center_y The y-coordinate of the pixel beam center in the image
 * @param pixel_size_x The pixel size of the detector in the x-direction in m
 * @param pixel_size_y The pixel size of the detector in the y-direction in m
 * @return The calculated distance from the beam center in m
*/
__device__ float get_distance_from_center(float x,
                                          float y,
                                          float center_x,
                                          float center_y,
                                          float pixel_size_x,
                                          float pixel_size_y) {
    /*
     * Since this calculation is for a broad, general exclusion, we can
     * use basic Pythagoras to calculate the distance from the center.
     * 
     * We add 0.5 in order to move to the center of the pixel. Since
     * starting at 0, 0, for instance, would infact be the corner.
    */
    float dx = ((x + 0.5f) - center_x) * pixel_size_x;
    float dy = ((y + 0.5f) - center_y) * pixel_size_y;

    return sqrtf(dx * dx + dy * dy);
}

/**
 * @brief Function to calculate the interplanar distance of a reflection.
 * The interplanar distance is calculated using the formula:
 *         d = λ / (2 * sin(ϴ))
 * @param wavelength The wavelength of the X-ray beam in Å
 * @param distance_to_detector The distance from the sample to the detector in m
 * @param distance_from_center The distance of the reflection from the beam center in m
 * @return The calculated d value
*/
__device__ float get_resolution(float wavelength,
                                float distance_to_detector,
                                float distance_from_center) {
    /*
     * Since the angle calculated is, in fact, 2ϴ, we halve to get the
     * proper value of ϴ
    */
    float theta = 0.5 * atanf(distance_from_center / distance_to_detector);
    return wavelength / (2 * sinf(theta));
}
#pragma endregion Res Mask Functions

#pragma region Res Mask Kernel
/**
 * @brief CUDA kernel to apply a resolution mask for an image.
 *
 * This kernel calculates the resolution for each pixel in an image based on the
 * distance from the beam center and the detector properties. It then masks out
 * pixels whose resolution falls outside the specified range [dmin, dmax],
 * provided that the pixel is not already masked, by setting the mask value of
 * the pixel to 0 in the mask data.
 *
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param mask_pitch The pitch of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param wavelength The wavelength of the X-ray beam in Ångströms.
 * @param distance_to_detector The distance from the sample to the detector in m.
 * @param beam_center_x The x-coordinate of the beam center in the image.
 * @param beam_center_y The y-coordinate of the beam center in the image.
 * @param pixel_size_x The pixel size of the detector in the x-direction in m.
 * @param pixel_size_y The pixel size of the detector in the y-direction in m.
 * @param dmin The minimum resolution (d-spacing) threshold.
 * @param dmax The maximum resolution (d-spacing) threshold.
 */
__global__ void apply_resolution_mask(uint8_t *mask,
                                      size_t mask_pitch,
                                      int width,
                                      int height,
                                      float wavelength,
                                      float distance_to_detector,
                                      float beam_center_x,
                                      float beam_center_y,
                                      float pixel_size_x,
                                      float pixel_size_y,
                                      float dmin,
                                      float dmax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width || y > height) return;  // Out of bounds

    if (mask[y * mask_pitch + x] == MASKED_PIXEL) {  // Check if the pixel is masked
        /*
        * If the pixel is already masked, we don't need to calculate the
        * resolution for it, so we can just leave it masked
        */
        return;
    }

    float distance_from_center = get_distance_from_center(
      x, y, beam_center_x, beam_center_y, pixel_size_x, pixel_size_y);
    float resolution =
      get_resolution(wavelength, distance_to_detector, distance_from_center);

    // Check if dmin is set and if the resolution is below it
    if (dmin > 0 && resolution < dmin) {
        mask[y * mask_pitch + x] = MASKED_PIXEL;
        return;
    }

    // Check if dmax is set and if the resolution is above it
    if (dmax > 0 && resolution > dmax) {
        mask[y * mask_pitch + x] = MASKED_PIXEL;
        return;
    }

    // If the pixel is not masked and the resolution is within the limits, set the resolution mask to 1
    mask[y * mask_pitch + x] = VALID_PIXEL;
    // ⛔🧊
}

/**
 * @brief Host function to launch the apply_resolution_mask kernel.
 *
 * This function sets up the kernel execution parameters and launches the
 * apply_resolution_mask kernel to generate and apply a resolution mask
 * onto the base mask for the detector.
 *
 * @param blocks The dimensions of the grid of blocks.
 * @param threads The dimensions of the grid of threads within each block.
 * @param shared_memory The size of shared memory required per block (in bytes).
 * @param stream The CUDA stream to execute the kernel.
 * @param mask Device pointer to the mask data indicating valid pixels.
 * @param params The parameters required to calculate the resolution mask.  
 */
void call_apply_resolution_mask(dim3 blocks,
                                dim3 threads,
                                size_t shared_memory,
                                cudaStream_t stream,
                                uint8_t *mask,
                                ResolutionMaskParams params) {
    // Launch the kernel
    apply_resolution_mask<<<blocks, threads, shared_memory, stream>>>(
      mask,
      params.mask_pitch,
      params.width,
      params.height,
      params.wavelength,
      params.detector.distance,
      params.detector.beam_center_x,
      params.detector.beam_center_y,
      params.detector.pixel_size_x,
      params.detector.pixel_size_y,
      params.dmin,
      params.dmax);
}
#pragma endregion Res Mask Kernel

#pragma region Spotfinding Functions
/**
 * @brief Calculate the sum, sum of squares, and count of valid pixels in the neighborhood.
 * @param image Device pointer to the image data.
 * @param mask Device pointer to the mask data indicating valid pixels.
 * @param background_mask (Optional) Device pointer to the background mask data. If nullptr, all pixels are considered for background calculation.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param x The x-coordinate of the current pixel.
 * @param y The y-coordinate of the current pixel.
 * @param kernel_width The half-width of the kernel (kernel size in x-direction).
 * @param kernel_height The half-height of the kernel (kernel size in y-direction).
 * @param sum (Output) The sum of the valid pixels in the neighborhood.
 * @param sumsq (Output) The sum of squares of the valid pixels in the neighborhood.
 * @param n (Output) The count of valid pixels in the neighborhood.
 */
__device__ void calculate_sums(pixel_t *image,
                               uint8_t *mask,
                               uint8_t *background_mask,
                               size_t image_pitch,
                               size_t mask_pitch,
                               int width,
                               int height,
                               int x,
                               int y,
                               uint8_t kernel_width,
                               uint8_t kernel_height,
                               uint &sum,
                               size_t &sumsq,
                               uint8_t &n) {
    sum = 0;
    sumsq = 0;
    n = 0;

    for (int row = max(0, y - kernel_height); row < min(y + kernel_height + 1, height);
         ++row) {
        int row_offset = image_pitch * row;
        int mask_offset = mask_pitch * row;
        for (int col = max(0, x - kernel_width); col < min(x + kernel_width + 1, width);
             ++col) {
            pixel_t pixel = image[row_offset + col];
            uint8_t mask_pixel = mask[mask_offset + col];
            bool include_pixel = mask_pixel != 0;  // If the pixel is valid
            if (background_mask != nullptr) {
                uint8_t background_mask_pixel = background_mask[mask_offset + col];
                include_pixel =
                  include_pixel
                  && (background_mask_pixel
                      == VALID_PIXEL);  // And is NOT a survivor from the erosion process
            }
            if (include_pixel) {
                sum += pixel;
                sumsq += pixel * pixel;
                n += 1;
            }
        }
    }
}

/**
 * @brief Determine if the current pixel is a strong pixel.
 * @param sum The sum of the valid pixels in the neighborhood.
 * @param sumsq The sum of squares of the valid pixels in the neighborhood.
 * @param n The count of valid pixels in the neighborhood.
 * @param this_pixel The intensity value of the current pixel.
 * @return True if the current pixel is a strong pixel, false otherwise.
 */
__device__ bool is_strong_pixel(uint sum, size_t sumsq, uint8_t n, pixel_t this_pixel) {
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

    return not_background && is_signal;
}
#pragma endregion Spotfinding Functions

#pragma region Spotfinding Kernels
/**
 * @brief CUDA kernel to perform spotfinding using a dispersion-based algorithm.
 * 
 * This kernel identifies strong pixels in the image based on analysis of the pixel neighborhood.
 * 
 * @param image Device pointer to the image data.
 * @param image_pitch The pitch of the image data.
 * @param mask Device pointer to the mask data indicating valid pixels.
 * @param background_mask (Optional) Device pointer to the background mask data. If nullptr, all pixels are considered for background calculation.
 * @param mask_pitch The pitch of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param kernel_width The radius of the kernel in the x-direction.
 * @param kernel_height The radius of the kernel in the y-direction.
 * @param result_strong (Output) Device pointer for the strong pixel mask data to be written to.
 */
__global__ void do_spotfinding_dispersion(pixel_t *image,
                                          size_t image_pitch,
                                          uint8_t *mask,
                                          uint8_t *background_mask,
                                          size_t mask_pitch,
                                          int width,
                                          int height,
                                          pixel_t max_valid_pixel_value,
                                          uint8_t kernel_width,
                                          uint8_t kernel_height,
                                          uint8_t *result_strong) {
    image = image + (image_pitch * height * blockIdx.z);
    // result_sum = result_sum + (image_pitch * height * blockIdx.z);
    // result_sumsq = result_sumsq + (image_pitch * height * blockIdx.z);
    // result_n = result_n + (mask_pitch * height * blockIdx.z);
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
    pixel_t this_pixel = image[y * image_pitch + x];
    bool px_is_valid =
      mask[y * mask_pitch + x] != 0 && this_pixel <= max_valid_pixel_value;

    if (px_is_valid) {
        calculate_sums(image,
                       mask,
                       background_mask,
                       image_pitch,
                       mask_pitch,
                       width,
                       height,
                       x,
                       y,
                       kernel_width,
                       kernel_height,
                       sum,
                       sumsq,
                       n);
    }

    if (x < width && y < height) {
        // result_sum[x + image_pitch * y] = sum;
        // result_sumsq[x + image_pitch * y] = sumsq;
        // result_n[x + mask_pitch * y] = n;

        // Calculate the thresholding
        if (px_is_valid && n > 1) {
            bool is_strong_pixel_flag = is_strong_pixel(sum, sumsq, n, this_pixel);
            result_strong[x + mask_pitch * y] = is_strong_pixel_flag;
        } else {
            result_strong[x + mask_pitch * y] = 0;
        }
    }
}
#pragma endregion Spotfinding Kernel

#pragma region Launch Wrappers
/**
 * @brief Wrapper function to call the dispersion-based spotfinding algorithm.
 * This function launches the `compute_dispersion_threshold_kernel` to perform
 * the spotfinding based on the basic dispersion threshold.
 *
 * @param blocks The dimensions of the grid of blocks.
 * @param threads The dimensions of the grid of threads within each block.
 * @param shared_memory The size of shared memory required per block (in bytes).
 * @param stream The CUDA stream to execute the kernel.
 * @param image PitchedMalloc object for the image data.
 * @param mask PitchedMalloc object for the mask data indicating valid pixels.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param result_strong (Output) Device pointer for the strong pixel mask data to be written to.
 * @param min_count The minimum number of valid pixels required in the local neighborhood. Default is 3.
 * @param n_sig_b The background noise significance level. Default is 6.0.
 * @param n_sig_s The signal significance level. Default is 3.0.
 */
void call_do_spotfinding_dispersion(dim3 blocks,
                                    dim3 threads,
                                    size_t shared_memory,
                                    cudaStream_t stream,
                                    PitchedMalloc<pixel_t> &image,
                                    PitchedMalloc<uint8_t> &mask,
                                    int width,
                                    int height,
                                    pixel_t max_valid_pixel_value,
                                    PitchedMalloc<uint8_t> *result_strong,
                                    uint8_t min_count,
                                    float n_sig_b,
                                    float n_sig_s) {
    /// One-direction width of kernel. Total kernel span is (K * 2 + 1)
    constexpr uint8_t basic_kernel_radius = 3;

    // Launch the dispersion threshold kernel
    compute_threshold_kernel<<<blocks, threads, shared_memory, stream>>>(
      image.get(),            // Image data pointer
      mask.get(),             // Mask data pointer
      result_strong->get(),   // Output mask pointer
      image.pitch,            // Image pitch
      mask.pitch,             // Mask pitch
      result_strong->pitch,   // Output mask pitch
      width,                  // Image width
      height,                 // Image height
      max_valid_pixel_value,  // Maximum valid pixel value
      basic_kernel_radius,    // Kernel width
      basic_kernel_radius,    // Kernel height
      min_count,              // Minimum count
      n_sig_b,                // Background significance level
      n_sig_s                 // Signal significance level
    );

    // do_spotfinding_dispersion<<<blocks, threads, shared_memory, stream>>>(
    //   image.get(),
    //   image.pitch,
    //   mask.get(),
    //   nullptr,  // No background mask
    //   mask.pitch,
    //   width,
    //   height,
    //   max_valid_pixel_value,
    //   basic_kernel_radius,
    //   basic_kernel_radius,
    //   result_strong->get());

    cudaStreamSynchronize(
      stream);  // Synchronize the CUDA stream to ensure the kernel is complete
}

/**
 * @brief Wrapper function to call the extended dispersion-based spotfinding algorithm.
 * This function launches the `compute_final_threshold_kernel` for final thresholding
 * after applying the dispersion mask and the `compute_dispersion_threshold_kernel`
 * for initial thresholding.
 *
 * @param blocks The dimensions of the grid of blocks.
 * @param threads The dimensions of the grid of threads within each block.
 * @param shared_memory The size of shared memory required per block (in bytes).
 * @param stream The CUDA stream to execute the kernel.
 * @param image PitchedMalloc object for the image data.
 * @param mask PitchedMalloc object for the mask data indicating valid pixels.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param result_strong (Output) Device pointer for the strong pixel mask data to be written to.
 * @param do_writeout Flag to indicate if the output should be written to file. Default is false.
 * @param min_count The minimum number of valid pixels required in the local neighborhood. Default is 3.
 * @param n_sig_b The background noise significance level. Default is 6.0.
 * @param n_sig_s The signal significance level. Default is 3.0.
 * @param threshold The global threshold for intensity values. Default is 10.0.
 */
void call_do_spotfinding_extended(dim3 blocks,
                                  dim3 threads,
                                  size_t shared_memory,
                                  cudaStream_t stream,
                                  PitchedMalloc<pixel_t> &image,
                                  PitchedMalloc<uint8_t> &mask,
                                  int width,
                                  int height,
                                  pixel_t max_valid_pixel_value,
                                  PitchedMalloc<uint8_t> *result_strong,
                                  bool do_writeout,
                                  uint8_t min_count,
                                  float n_sig_b,
                                  float n_sig_s,
                                  float threshold) {
    // Allocate intermediate buffer for the dispersion mask on the device
    PitchedMalloc<uint8_t> d_dispersion_mask(width, height);

    constexpr uint8_t first_pass_kernel_radius = 3;

    /*
     * First pass
     * Perform the initial dispersion thresholding only on the background
     * threshold. The surviving pixels are then used as a mask later to
     * exclude them from the background calculation in the second pass.
    */
    {
        printf("First pass\n");
        // First pass: Perform the initial dispersion thresholding
        compute_dispersion_threshold_kernel<<<blocks, threads, shared_memory, stream>>>(
          image.get(),               // Image data pointer
          mask.get(),                // Mask data pointer
          d_dispersion_mask.get(),   // Output dispersion mask pointer
          image.pitch,               // Image pitch
          mask.pitch,                // Mask pitch
          d_dispersion_mask.pitch,   // Output dispersion mask pitch
          width,                     // Image width
          height,                    // Image height
          max_valid_pixel_value,     // Maximum valid pixel value
          first_pass_kernel_radius,  // Kernel radius
          first_pass_kernel_radius,  // Kernel radius
          min_count,                 // Minimum count
          n_sig_b,                   // Background significance level
          n_sig_s                    // Signal significance level
        );
        cudaStreamSynchronize(
          stream);  // Synchronize the CUDA stream to ensure the first pass is complete

        printf("First pass complete\n");
        // Optional: Write out the first pass result if needed
        if (do_writeout) {
            // Write to PNG
            {
                // Function to transform the pixel values: if non-zero, set to 255, otherwise set to 0
                auto convert_pixel = [](uint8_t pixel) -> uint8_t {
                    // return pixel ? 255 : 0;
                    if (pixel == MASKED_PIXEL) {
                        return 0;
                    } else {  // if (pixel == VALID_PIXEL)
                        return 255;
                    }
                };

                // Usage in your existing code
                save_device_data_to_png(
                  d_dispersion_mask.get(),          // Device pointer to the 2D array
                  d_dispersion_mask.pitch_bytes(),  // Device pitch in bytes
                  width,                            // Width of the image
                  height,                           // Height of the image
                  stream,                           // CUDA stream
                  "first_pass_dispersion_result",   // Output filename
                  convert_pixel                     // Pixel transformation function
                );
            }
            // Write to TXT
            {
                auto is_valid_pixel = [](uint8_t pixel) { return pixel != 0; };

                save_device_data_to_txt(
                  d_dispersion_mask.get(),          // Device pointer to the 2D array
                  d_dispersion_mask.pitch_bytes(),  // Device pitch in bytes
                  width,                            // Width of the image
                  height,                           // Height of the image
                  stream,                           // CUDA stream
                  "first_pass_dispersion_result",   // Output filename
                  is_valid_pixel                    // Pixel condition function
                );
            }
        }
    }

    /*
     * Erosion pass
     * Erode the first pass results.
     * The surviving pixels are then used as a mask to exclude them
     * from the background calculation in the second pass.
    */
    PitchedMalloc<uint8_t> d_erosion_mask(width, height);
    {
        dim3 threads_per_erosion_block(32, 32);
        dim3 erosion_blocks(
          (width + threads_per_erosion_block.x - 1) / threads_per_erosion_block.x,
          (height + threads_per_erosion_block.y - 1) / threads_per_erosion_block.y);

        // Calculate the shared memory size for the erosion kernel
        size_t erosion_shared_memory =
          (threads_per_erosion_block.x + 2 * first_pass_kernel_radius)
          * (threads_per_erosion_block.y + 2 * first_pass_kernel_radius)
          * sizeof(uint8_t);

        // Perform erosion
        erosion_kernel<<<erosion_blocks,
                         threads_per_erosion_block,
                         erosion_shared_memory,
                         stream>>>(d_dispersion_mask.get(),
                                   d_erosion_mask.get(),
                                   mask.get(),
                                   d_dispersion_mask.pitch,
                                   d_erosion_mask.pitch,
                                   mask.pitch,
                                   width,
                                   height,
                                   first_pass_kernel_radius);
        cudaStreamSynchronize(stream);

        // Print the erosion result if needed
        if (do_writeout) {
            // Write to PNG
            {
                auto show_masked = [](uint8_t pixel) -> uint8_t {
                    if (pixel == MASKED_PIXEL) {
                        return 0;
                    } else {  // if (pixel == VALID_PIXEL)
                        return 255;
                    }
                };

                save_device_data_to_png(
                  d_erosion_mask.get(),          // Device pointer to the 2D array
                  d_erosion_mask.pitch_bytes(),  // Device pitch in bytes
                  width,                         // Width of the image
                  height,                        // Height of the image
                  stream,                        // CUDA stream
                  "eroded_dispersion_result",    // Output filename
                  show_masked                    // Pixel transformation function
                );
            }
            // Write to TXT
            {
                auto is_masked_pixel = [](uint8_t pixel) {
                    return pixel == MASKED_PIXEL;
                };

                save_device_data_to_txt(
                  d_erosion_mask.get(),          // Device pointer to the 2D array
                  d_erosion_mask.pitch_bytes(),  // Device pitch in bytes
                  width,                         // Width of the image
                  height,                        // Height of the image
                  stream,                        // CUDA stream
                  "eroded_dispersion_result",    // Output filename
                  is_masked_pixel                // Pixel condition function
                );
            }
        }
    }

    constexpr uint8_t second_pass_kernel_radius = 5;

    /*
     * Second pass
     * Perform the final thresholding using the dispersion mask.
    */
    {
        printf("Second pass\n");
        // Second pass: Perform the final thresholding using the dispersion mask
        compute_final_threshold_kernel<<<blocks, threads, shared_memory, stream>>>(
          image.get(),                // Image data pointer
          mask.get(),                 // Mask data pointer
          d_erosion_mask.get(),       // Dispersion mask pointer
          result_strong->get(),       // Output result mask pointer
          image.pitch,                // Image pitch
          mask.pitch,                 // Mask pitch
          d_erosion_mask.pitch,       // Dispersion mask pitch
          result_strong->pitch,       // Output result mask pitch
          width,                      // Image width
          height,                     // Image height
          max_valid_pixel_value,      // Maximum valid pixel value
          second_pass_kernel_radius,  // Kernel radius
          second_pass_kernel_radius,  // Kernel radius
          n_sig_s,                    // Signal significance level
          threshold                   // Global threshold
        );
        cudaStreamSynchronize(
          stream);  // Synchronize the CUDA stream to ensure the second pass is complete

        printf("Second pass complete\n");
        // Optional: Write out the final result if needed
        if (do_writeout) {
            auto convert_pixel = [](uint8_t pixel) -> uint8_t {
                if (pixel == VALID_PIXEL) {
                    return 255;
                } else {
                    return 0;
                }
            };

            save_device_data_to_png(
              result_strong->get(),               // Device pointer to the 2D array
              mask.pitch_bytes(),                 // Device pitch in bytes
              width,                              // Width of the image
              height,                             // Height of the image
              stream,                             // CUDA stream
              "final_extended_threshold_result",  // Output filename
              convert_pixel                       // Pixel transformation function
            );

            auto is_valid_pixel = [](uint8_t pixel) { return pixel != 0; };

            save_device_data_to_txt(
              result_strong->get(),               // Device pointer to the 2D array
              mask.pitch_bytes(),                 // Device pitch in bytes
              width,                              // Width of the image
              height,                             // Height of the image
              stream,                             // CUDA stream
              "final_extended_threshold_result",  // Output filename
              is_valid_pixel                      // Pixel condition function
            );
        }
    }
}

#pragma endregion Launch Wrappers
