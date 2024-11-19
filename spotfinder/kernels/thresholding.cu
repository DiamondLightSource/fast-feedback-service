/**
 * @file thresholding.cu
 * @brief Contains CUDA kernel implementations for thresholding image 
 *        data.
 *
 * This file contains CUDA kernels for thresholding image data based on
 * the variance and mean of the local neighbourhood of each pixel. These
 * thresholds are used to identify potential signal spots against a
 * background in the input image based on local variance, mean, and
 * user-defined significance levels.
 * 
 * @note The __restrict__ keyword is used to indicate to the compiler
 *       that the two pointers are not aliased, allowing the compiler to
 *       perform more aggressive optimizations.
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cuda/std/tuple>

#include "cuda_common.hpp"
#include "device_common.cuh"
#include "thresholding.cuh"

namespace cg = cooperative_groups;

#pragma region Global constants
/*
 * Constants for kernels
 * extern keyword is used to declare a variable that is defined in
 * another file. This links the constant global variable to the
 * kernel_constants variable defined in spotfinder.cu
 */
extern __constant__ KernelConstants kernel_constants;
#pragma endregion Global constants

#pragma region Device Functions
/**
 * @brief Calculate the dispersion flags for a given pixel.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param this_pixel The pixel value at the current position.
 * @param x The x-coordinate of the pixel.
 * @param y The y-coordinate of the pixel.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param kernel_width The radius of the kernel in the x-direction.
 * @param kernel_height The radius of the kernel in the y-direction.
 * @param min_count Minimum number of valid pixels required to be considered a valid spot.
 * @param n_sig_b Background noise significance level.
 * @param n_sig_s Signal significance level.
 * @return A tuple <bool not_background, bool is_signal, uint8_t n> where:
 *        - not_background: True if the pixel is not part of the background.
 *        - is_signal: True if the pixel is a strong signal.
 *        - n: The number of valid pixels in the local neighbourhood.
 */
__device__ cuda::std::tuple<bool, bool, uint8_t> calculate_dispersion_flags(
  PitchedArray2D<pixel_t> image,
  PitchedArray2D<uint8_t> mask,
  pixel_t this_pixel,
  int x,
  int y,
  int width,
  int height,
  uint8_t kernel_radius,
  uint8_t min_count,
  float n_sig_b,
  float n_sig_s) {
    // Initialize variables for computing the local sum and count
    uint sum = 0;
    size_t sumsq = 0;
    uint8_t n = 0;

    int row_start = max(0, y - kernel_radius);
    int row_end = min(y + kernel_radius + 1, height);

    for (int row = row_start; row < row_end; ++row) {
        int col_start = max(0, x - kernel_radius);
        int col_end = min(x + kernel_radius + 1, width);

        for (int col = col_start; col < col_end; ++col) {
            pixel_t pixel = image(col, row);
            uint8_t mask_pixel = mask(col, row);
            bool include_pixel = mask_pixel != 0;  // If the pixel is valid
            if (include_pixel) {
                sum += pixel;
                sumsq += pixel * pixel;
                n += 1;
            }
        }
    }

    // Compute local mean and variance
    float sum_f = static_cast<float>(sum);
    float sumsq_f = static_cast<float>(sumsq);

    float mean = sum_f / n;
    float variance = (n * sumsq_f - (sum_f * sum_f)) / (n * (n - 1));
    float dispersion = variance / mean;

    // Compute the background threshold and signal threshold
    float background_threshold = 1 + n_sig_b * sqrt(2.0f / (n - 1));
    bool not_background = dispersion > background_threshold;
    float signal_threshold = mean + n_sig_s * sqrt(mean);

    // Check if the pixel is a strong pixel
    bool is_signal = this_pixel > signal_threshold;

    return {not_background, is_signal, n};
}
#pragma endregion

#pragma region Thresholding kernels
/**
 * @brief Kernel for computing the basic threshold based on variance and mean.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param result_mask Pointer to the output mask data where results will be stored.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param result_pitch The pitch of the result mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param min_count Minimum number of valid pixels required to be considered a valid spot.
 * @param threshold The global threshold for intensity.
 * @param n_sig_b Background noise significance level.
 * @param n_sig_s Signal significance level.
 */
__global__ void dispersion(pixel_t __restrict__ *image_ptr,
                           uint8_t __restrict__ *mask_ptr,
                           uint8_t __restrict__ *result_mask_ptr) {
    // Move pointers to the correct slice
    image_ptr =
      image_ptr + (kernel_constants.image_pitch * kernel_constants.height * blockIdx.z);
    result_mask_ptr =
      result_mask_ptr
      + (kernel_constants.mask_pitch * kernel_constants.height * blockIdx.z);

    // Create pitched arrays for data access
    PitchedArray2D<pixel_t> image(image_ptr, &kernel_constants.image_pitch);
    PitchedArray2D<uint8_t> mask(mask_ptr, &kernel_constants.mask_pitch);
    PitchedArray2D<uint8_t> result_mask(result_mask_ptr,
                                        &kernel_constants.result_pitch);

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (x >= kernel_constants.width || y >= kernel_constants.height)
        return;  // Out of bounds guard

    pixel_t this_pixel = image(x, y);

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid =
      mask(x, y) != 0 && this_pixel <= kernel_constants.max_valid_pixel_value;

    // Validity guard
    if (!px_is_valid) {
        result_mask(x, y) = 0;
        return;
    }

    // Calculate the dispersion flags
    auto [not_background, is_signal, n] =
      calculate_dispersion_flags(image,
                                 mask,
                                 this_pixel,
                                 x,
                                 y,
                                 kernel_constants.width,
                                 kernel_constants.height,
                                 KERNEL_RADIUS,
                                 kernel_constants.min_count,
                                 kernel_constants.n_sig_b,
                                 kernel_constants.n_sig_s);

    result_mask(x, y) = not_background && is_signal && n > 1;
}

/**
 * @brief Kernel for computing the dispersion threshold.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param result_mask Pointer to the output mask data where results will be stored.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param result_pitch The pitch of the result mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param kernel_width The radius of the kernel in the x-direction.
 * @param kernel_height The radius of the kernel in the y-direction.
 * @param min_count Minimum number of valid pixels required to be considered a valid spot.
 * @param n_sig_b Background noise significance level.
 * @param n_sig_s Signal significance level.
 */
__global__ void dispersion_extended_first_pass(pixel_t __restrict__ *image_ptr,
                                               uint8_t __restrict__ *mask_ptr,
                                               uint8_t __restrict__ *result_mask_ptr) {
    // Move pointers to the correct slice
    image_ptr =
      image_ptr + (kernel_constants.image_pitch * kernel_constants.height * blockIdx.z);
    result_mask_ptr =
      result_mask_ptr
      + (kernel_constants.mask_pitch * kernel_constants.height * blockIdx.z);

    // Create pitched arrays for data access
    PitchedArray2D<pixel_t> image(image_ptr, &kernel_constants.image_pitch);
    PitchedArray2D<uint8_t> mask(mask_ptr, &kernel_constants.mask_pitch);
    PitchedArray2D<uint8_t> result_mask(result_mask_ptr,
                                        &kernel_constants.result_pitch);

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (x >= kernel_constants.width || y >= kernel_constants.height)
        return;  // Out of bounds guard

    pixel_t this_pixel = image(x, y);

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid =
      mask(x, y) != 0 && this_pixel <= kernel_constants.max_valid_pixel_value;

    // Validity guard
    if (!px_is_valid) {
        result_mask(x, y) = 0;
        return;
    }

    // Calculate the dispersion flags
    auto [not_background, is_signal, n] =
      calculate_dispersion_flags(image,
                                 mask,
                                 this_pixel,
                                 x,
                                 y,
                                 kernel_constants.width,
                                 kernel_constants.height,
                                 KERNEL_RADIUS,
                                 kernel_constants.min_count,
                                 kernel_constants.n_sig_b,
                                 kernel_constants.n_sig_s);

    result_mask(x, y) = not_background && n > 1;
}

/**
 * @brief Kernel for computing the final threshold after dispersion mask.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param dispersion_mask Pointer to the dispersion mask used for extended algorithm.
 * @param result_mask Pointer to the output mask data where results will be stored.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param dispersion_mask_pitch The pitch of the dispersion mask data.
 * @param result_mask_pitch The pitch of the result mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param n_sig_s Signal significance level.
 * @param threshold Global threshold for the intensity.
 */
__global__ void dispersion_extended_second_pass(
  pixel_t __restrict__ *image_ptr,
  uint8_t __restrict__ *mask_ptr,
  uint8_t __restrict__ *dispersion_mask_ptr,
  uint8_t __restrict__ *result_mask_ptr,
  size_t dispersion_mask_pitch) {
    // Move pointers to the correct slice
    image_ptr =
      image_ptr + (kernel_constants.image_pitch * kernel_constants.height * blockIdx.z);
    result_mask_ptr =
      result_mask_ptr
      + (kernel_constants.result_pitch * kernel_constants.height * blockIdx.z);

    // Create pitched arrays for data access
    PitchedArray2D<pixel_t> image(image_ptr, &kernel_constants.image_pitch);
    PitchedArray2D<uint8_t> mask(mask_ptr, &kernel_constants.mask_pitch);
    PitchedArray2D<uint8_t> dispersion_mask(dispersion_mask_ptr,
                                            &dispersion_mask_pitch);
    PitchedArray2D<uint8_t> result_mask(result_mask_ptr,
                                        &kernel_constants.result_pitch);

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (x >= kernel_constants.width || y >= kernel_constants.height)
        return;  // Out of bounds guard

    pixel_t this_pixel = image(x, y);

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid =
      mask(x, y) != 0 && this_pixel <= kernel_constants.max_valid_pixel_value;

    // Initialize variables for computing the local sum and count
    uint sum = 0;
    uint8_t n = 0;

    int row_start = max(0, y - KERNEL_RADIUS_EXTENDED);
    int row_end = min(y + KERNEL_RADIUS_EXTENDED + 1, kernel_constants.height);

    for (int row = row_start; row < row_end; ++row) {
        int col_start = max(0, x - KERNEL_RADIUS_EXTENDED);
        int col_end = min(x + KERNEL_RADIUS_EXTENDED + 1, kernel_constants.width);

        for (int col = col_start; col < col_end; ++col) {
            pixel_t pixel = image(col, row);
            uint8_t mask_pixel = mask(col, row);
            uint8_t disp_mask_pixel = dispersion_mask(col, row);
            /*
               * Check if the pixel is valid. That means that it is not
               * masked and was not marked as potentially signal in the
               * dispersion mask.
              */
            bool include_pixel =
              mask_pixel != MASKED_PIXEL && disp_mask_pixel != MASKED_PIXEL;
            if (include_pixel) {
                sum += pixel;
                n += 1;
            }
        }
    }

    constexpr float threshold = 0.0f;  // DIALS default value

    // Calculate the thresholding
    if (px_is_valid && n > 0) {
        float sum_f = static_cast<float>(sum);

        // The pixel must have been marked as potentially signal in the dispersion mask
        bool disp_mask = dispersion_mask(x, y) == MASKED_PIXEL;
        // The pixel must be above the global threshold
        bool global_mask = image(x, y) > threshold;
        // Calculate the local mean
        float mean = (n > 1 ? sum_f / n : 0);  // If n is less than 1, set mean to 0
        // The pixel must be above the local threshold
        bool local_mask =
          image(x, y) >= (mean + kernel_constants.n_sig_s * sqrtf(mean));

        result_mask(x, y) = disp_mask && global_mask && local_mask;
    } else {
        result_mask(x, y) = 0;
    }
}
#pragma endregion