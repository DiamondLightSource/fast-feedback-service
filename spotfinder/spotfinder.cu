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
#include "spotfinder.cuh"

namespace cg = cooperative_groups;

#pragma region Res Mask Functions
/**
 * @brief Function to calculate the distance of a pixel from the beam center.
 * @param x The x-coordinate of the pixel in the image
 * @param y The y-coordinate of the pixel in the image
 * @param center_x The x-coordinate of the pixel beam center in the image
 * @param center_y The y-coordinate of the pixel beam center in the image
 * @param pixel_size_x The pixel size of the detector in the x-direction in mm
 * @param pixel_size_y The pixel size of the detector in the y-direction in mm
 * @return The calculated distance from the beam center in mm
*/
__device__ float get_distance_from_centre(float x,
                                          float y,
                                          float centre_x,
                                          float centre_y,
                                          float pixel_size_x,
                                          float pixel_size_y) {
    /*
     * Since this calculation is for a broad, general exclusion, we can
     * use basic Pythagoras to calculate the distance from the center.
    */
    float dx = (x - centre_x) * pixel_size_x;
    float dy = (y - centre_y) * pixel_size_y;
    return sqrtf(dx * dx + dy * dy);
}

/**
 * @brief Function to calculate the interplanar distance of a reflection.
 * The interplanar distance is calculated using the formula:
 *         d = λ / (2 * sin(ϴ))
 * @param wavelength The wavelength of the X-ray beam in Å
 * @param distance_to_detector The distance from the sample to the detector in mm
 * @param distance_from_center The distance of the reflection from the beam center in mm
 * @return The calculated d value
*/
__device__ float get_resolution(float wavelength,
                                float distance_to_detector,
                                float distance_from_centre) {
    /*
     * Since the angle calculated is, in fact, 2ϴ, we halve to get the
     * proper value of ϴ
    */
    float theta = 0.5 * atanf(distance_from_centre / distance_to_detector);
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
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param wavelength The wavelength of the X-ray beam in Ångströms.
 * @param distance_to_detector The distance from the sample to the detector in mm.
 * @param beam_center_x The x-coordinate of the beam center in the image.
 * @param beam_center_y The y-coordinate of the beam center in the image.
 * @param pixel_size_x The pixel size of the detector in the x-direction in mm.
 * @param pixel_size_y The pixel size of the detector in the y-direction in mm.
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

    float distance_from_centre = get_distance_from_centre(
      x, y, beam_center_x, beam_center_y, pixel_size_x, pixel_size_y);
    float resolution =
      get_resolution(wavelength, distance_to_detector, distance_from_centre);

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
 * @param image_pitch The pitch (width in bytes) of the image data.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param x The x-coordinate of the current pixel.
 * @param y The y-coordinate of the current pixel.
 * @param kernel_width The radius of the kernel in the x-direction.
 * @param kernel_height The radius of the kernel in the y-direction.
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
                               int kernel_width,
                               int kernel_height,
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

#pragma region Spotfinding Kernel
/**
 * @brief CUDA kernel to perform spotfinding using a dispersion-based algorithm.
 * 
 * This kernel identifies strong pixels in the image based on analysis of the pixel neighborhood.
 * 
 * @param image Device pointer to the image data.
 * @param image_pitch The pitch (width in bytes) of the image data.
 * @param mask Device pointer to the mask data indicating valid pixels.
 * @param background_mask (Optional) Device pointer to the background mask data. If nullptr, all pixels are considered for background calculation.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
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
                                          int kernel_width,
                                          int kernel_height,
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
                                    uint8_t *result_strong) {
    /// One-direction width of kernel. Total kernel span is (K_W * 2 + 1)
    constexpr int basic_kernel_width = 3;
    /// One-direction height of kernel. Total kernel span is (K_H * 2 + 1)
    constexpr int basic_kernel_height = 3;

    do_spotfinding_dispersion<<<blocks, threads, shared_memory, stream>>>(
      image.get(),
      image.pitch,
      mask.get(),
      nullptr,  // No background mask
      mask.pitch,
      width,
      height,
      max_valid_pixel_value,
      basic_kernel_width,
      basic_kernel_height,
      result_strong);
}

/**
 * @brief Wrapper function to call the extended spotfinding algorithm.
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
                                  uint8_t *result_strong,
                                  bool do_writeout) {
    // Allocate memory for the intermediate result buffer
    PitchedMalloc<uint8_t> d_result_strong_buffer(width, height);

    constexpr int first_pass_kernel_radius = 3;

    // Do the first step of spotfinding
    do_spotfinding_dispersion<<<blocks, threads, shared_memory, stream>>>(
      image.get(),
      image.pitch,
      mask.get(),
      nullptr,  // No background mask
      mask.pitch,
      width,
      height,
      max_valid_pixel_value,
      first_pass_kernel_radius,  // One-direction width of kernel. Total kernel span is (width * 2 + 1)
      first_pass_kernel_radius,  // One-direction height of kernel. Total kernel span is (height * 2 + 1)
      d_result_strong_buffer.get());
    cudaStreamSynchronize(stream);

    // Print the first pass result to png
    if (do_writeout) {
        auto buffer = std::vector<uint8_t>(width * height);
        cudaMemcpy2DAsync(buffer.data(),
                          width,
                          d_result_strong_buffer.get(),
                          d_result_strong_buffer.pitch_bytes(),
                          width,
                          height,
                          cudaMemcpyDeviceToHost,
                          stream);
        for (auto &pixel : buffer) {
            pixel = pixel ? 0 : 255;
        }
        lodepng::encode("first_pass_result.png",
                        reinterpret_cast<uint8_t *>(buffer.data()),
                        width,
                        height,
                        LCT_GREY);
    }

    /*
     * Allocate memory for the erosion mask. This is a mask of pixels that
     * are considered strong in the first pass, but were removed in the
     * upcoming erosion step.
    */
    PitchedMalloc<uint8_t> d_erosion_mask(width, height);

    {  // Get erosion results
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
                         stream>>>(d_result_strong_buffer.get(),
                                   d_erosion_mask.get(),
                                   d_erosion_mask.pitch_bytes(),
                                   width,
                                   height,
                                   first_pass_kernel_radius);
        cudaStreamSynchronize(stream);
    }
    if (do_writeout) {
        // Print the erosion mask to png
        auto mask_buffer = std::vector<uint8_t>(width * height);
        cudaMemcpy2DAsync(mask_buffer.data(),
                          width,
                          d_erosion_mask.get(),
                          d_erosion_mask.pitch_bytes(),
                          width,
                          height,
                          cudaMemcpyDeviceToHost,
                          stream);
        // Create an RGB buffer to store the image data
        auto image_mask =
          std::vector<std::array<uint8_t, 3>>(width * height, {0, 0, 0});

        for (int y = 0, k = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x, ++k) {
                image_mask[k] = {255, 0, 0};  // Default to white
                if (mask_buffer[k]) {
                    image_mask[k] = {
                      255,
                      255,
                      255};  // Set to red if the pixel is part of the erosion mask
                }
            }
        }
        lodepng::encode("erosion_mask.png",
                        reinterpret_cast<uint8_t *>(image_mask.data()),
                        width,
                        height,
                        LCT_RGB);
    }

    constexpr int second_pass_kernel_radius = 5;
    // Perform the second step of spotfinding
    do_spotfinding_dispersion<<<blocks, threads, shared_memory, stream>>>(
      image.get(),
      image.pitch,
      mask.get(),
      d_erosion_mask.get(),
      mask.pitch,
      width,
      height,
      max_valid_pixel_value,
      second_pass_kernel_radius,
      second_pass_kernel_radius,
      result_strong);
    cudaStreamSynchronize(stream);
}
#pragma endregion Launch Wrappers
