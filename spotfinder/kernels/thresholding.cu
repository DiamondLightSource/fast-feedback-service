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
 * @brief Reduce one pixel's neighbourhood to the moments the dispersion
 *        decision needs.
 *
 * Walks the (2r+1) * (2r+1) window around the pixel out of shared memory,
 * skipping neighbours that fall outside the image or that the mask excludes.
 * The sums are 64-bit so that a saturated neighbour cannot wrap them: the
 * centre pixel is the only one the trusted-range guard covers.
 *
 * @param image The block's image tile in shared memory, halo included.
 * @param mask The matching mask tile.
 * @param this_pixel The centre pixel's own count, in PHOTONS.
 * @param local_x The centre pixel's x-coordinate within the shared tile.
 * @param local_y The centre pixel's y-coordinate within the shared tile.
 * @param global_x The centre pixel's x-coordinate within the image.
 * @param global_y The centre pixel's y-coordinate within the image.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param kernel_radius One-direction width of the window.
 * @return The reduced moments, with n counting the valid pixels found.
 */
__device__ DispersionMoments calculate_dispersion_moments(PitchedArray2D<pixel_t> image,
                                                          PitchedArray2D<uint8_t> mask,
                                                          pixel_t this_pixel,
                                                          int local_x,
                                                          int local_y,
                                                          int global_x,
                                                          int global_y,
                                                          uint32_t width,
                                                          uint32_t height,
                                                          uint8_t kernel_radius) {
    uint64_t sum = 0;
    uint64_t sumsq = 0;
    uint8_t n = 0;

#pragma unroll
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
#pragma unroll
        for (int j = -kernel_radius; j <= kernel_radius; ++j) {
            // Guard against out-of-bounds access
            int gx = global_x + j;
            int gy = global_y + i;
            if (!in_bounds(gx, gy, width, height)) {
                continue;
            }

            // Calculate the local coordinates
            int lx = local_x + j;
            int ly = local_y + i;
            uint8_t mask_pixel = mask(lx, ly);
            if (mask_pixel) {
                const uint64_t pixel = image(lx, ly);
                sum += pixel;
                sumsq += pixel * pixel;
                ++n;
            }
        }
    }

    return {sum, sumsq, this_pixel, n};
}
#pragma endregion

#pragma region Thresholding kernels
/**
 * @brief The dispersion first pass, shared by both thresholding algorithms.
 *
 * Image dimensions, pitches and thresholds are read from kernel_constants.
 * blockIdx.z selects the image within a batch.
 *
 * @param image_ptr Pointer to the input image data.
 * @param mask_ptr Pointer to the mask data indicating valid pixels.
 * @param result_mask_ptr Pointer to the output mask where results are stored.
 * @param require_signal Whether the centre pixel must clear the signal
 *                       threshold as well as the background one; the extended
 *                       algorithm leaves that to its second pass.
 */
__device__ inline void dispersion_pass(pixel_t __restrict__ *image_ptr,
                                       uint8_t __restrict__ *mask_ptr,
                                       uint8_t __restrict__ *result_mask_ptr,
                                       bool require_signal) {
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
    int global_x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int global_y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (global_x >= kernel_constants.width || global_y >= kernel_constants.height)
        return;  // Out of bounds guard

    // Allocate shared memory for the image data
    extern __shared__ uint8_t shared_mem[];
    // Partition shared memory for image and mask data
    uint8_t *shared_mask_ptr = shared_mem;
    size_t shared_partition_size =
      (blockDim.x + KERNEL_RADIUS * 2) * (blockDim.y + KERNEL_RADIUS * 2);
    pixel_t *shared_image_ptr =
      reinterpret_cast<pixel_t *>(&shared_mask_ptr[shared_partition_size]);

    // Create pitched arrays for shared memory access
    size_t shared_pitch = blockDim.x + KERNEL_RADIUS * 2;
    PitchedArray2D<pixel_t> shared_image(shared_image_ptr, &shared_pitch);
    PitchedArray2D<uint8_t> shared_mask(shared_mask_ptr, &shared_pitch);

    int local_x = threadIdx.x + KERNEL_RADIUS;
    int local_y = threadIdx.y + KERNEL_RADIUS;

    // Get the pixel value at the current position
    pixel_t this_pixel = image(global_x, global_y);

    // Load this pixel into shared memory
    shared_image(local_x, local_y) = this_pixel;
    shared_mask(local_x, local_y) = mask(global_x, global_y);

    // Load the surrounding pixels into shared memory
    load_halo(block,
              global_x,
              global_y,
              kernel_constants.width,
              kernel_constants.height,
              KERNEL_RADIUS,
              KERNEL_RADIUS,
              cuda::std::make_tuple(image, shared_image),
              cuda::std::make_tuple(mask, shared_mask));

    // Sync threads to ensure all shared memory is loaded
    block.sync();

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid = shared_mask(local_x, local_y) != 0
                       && this_pixel <= kernel_constants.max_valid_pixel_value;

    // Validity guard
    if (!px_is_valid) {
        result_mask(global_x, global_y) = 0;
        return;
    }

    // Reduce the neighbourhood, then apply the shared thresholding decision
    const DispersionMoments moments =
      calculate_dispersion_moments(shared_image,
                                   shared_mask,
                                   this_pixel,
                                   local_x,
                                   local_y,
                                   global_x,
                                   global_y,
                                   kernel_constants.width,
                                   kernel_constants.height,
                                   KERNEL_RADIUS);
    const DispersionFlags flags = dispersion_flags(moments,
                                                   kernel_constants.min_count,
                                                   kernel_constants.n_sig_b,
                                                   kernel_constants.n_sig_s);

    result_mask(global_x, global_y) =
      flags.not_background && (flags.is_signal || !require_signal) && moments.n > 1;
}

/**
 * @brief Kernel for computing the basic threshold based on variance and mean.
 */
__global__ void dispersion(pixel_t __restrict__ *image_ptr,
                           uint8_t __restrict__ *mask_ptr,
                           uint8_t __restrict__ *result_mask_ptr) {
    dispersion_pass(image_ptr, mask_ptr, result_mask_ptr, true);
}

/**
 * @brief Kernel for the first pass of the extended dispersion threshold.
 */
__global__ void dispersion_extended_first_pass(pixel_t __restrict__ *image_ptr,
                                               uint8_t __restrict__ *mask_ptr,
                                               uint8_t __restrict__ *result_mask_ptr) {
    dispersion_pass(image_ptr, mask_ptr, result_mask_ptr, false);
}

/**
 * @brief Kernel for computing the final threshold after dispersion mask.
 *
 * Image dimensions, pitches and thresholds are read from kernel_constants.
 * blockIdx.z selects the image within a batch.
 *
 * @param image_ptr Pointer to the input image data.
 * @param mask_ptr Pointer to the mask data indicating valid pixels.
 * @param dispersion_mask_ptr Pointer to the eroded dispersion mask from the
 *        first pass.
 * @param result_mask_ptr Pointer to the output mask where results are stored.
 * @param dispersion_mask_pitch The pitch of the dispersion mask, in elements.
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
    int global_x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int global_y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (global_x >= kernel_constants.width || global_y >= kernel_constants.height)
        return;  // Out of bounds guard

    // Allocate shared memory for the image data
    extern __shared__ uint8_t shared_mem[];
    // Partition shared memory for image, mask and dispersion mask data
    uint8_t *shared_mask_ptr = shared_mem;
    size_t shared_partition_size = (blockDim.x + KERNEL_RADIUS_EXTENDED * 2)
                                   * (blockDim.y + KERNEL_RADIUS_EXTENDED * 2);
    uint8_t *shared_dispersion_mask_ptr = &shared_mask_ptr[shared_partition_size];
    pixel_t *shared_image_ptr =
      reinterpret_cast<pixel_t *>(&shared_dispersion_mask_ptr[shared_partition_size]);

    // Create pitched arrays for shared memory access
    size_t shared_pitch = blockDim.x + KERNEL_RADIUS_EXTENDED * 2;
    PitchedArray2D<pixel_t> shared_image(shared_image_ptr, &shared_pitch);
    PitchedArray2D<uint8_t> shared_mask(shared_mask_ptr, &shared_pitch);
    PitchedArray2D<uint8_t> shared_dispersion_mask(shared_dispersion_mask_ptr,
                                                   &shared_pitch);

    int local_x = threadIdx.x + KERNEL_RADIUS_EXTENDED;
    int local_y = threadIdx.y + KERNEL_RADIUS_EXTENDED;

    pixel_t this_pixel = image(global_x, global_y);

    // Load this pixel into shared memory
    shared_image(local_x, local_y) = this_pixel;
    shared_mask(local_x, local_y) = mask(global_x, global_y);
    shared_dispersion_mask(local_x, local_y) = dispersion_mask(global_x, global_y);

    // Load the surrounding pixels into shared memory
    load_halo(block,
              global_x,
              global_y,
              kernel_constants.width,
              kernel_constants.height,
              KERNEL_RADIUS_EXTENDED,
              KERNEL_RADIUS_EXTENDED,
              cuda::std::make_tuple(image, shared_image),
              cuda::std::make_tuple(mask, shared_mask),
              cuda::std::make_tuple(dispersion_mask, shared_dispersion_mask));

    // Sync threads to ensure all shared memory is loaded
    block.sync();

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid = shared_mask(local_x, local_y) != 0
                       && this_pixel <= kernel_constants.max_valid_pixel_value;

    // Initialize variables for computing the local sum and count
    uint64_t sum = 0;
    uint8_t n = 0;

#pragma unroll
    for (int i = -KERNEL_RADIUS_EXTENDED; i <= KERNEL_RADIUS_EXTENDED; ++i) {
#pragma unroll
        for (int j = -KERNEL_RADIUS_EXTENDED; j <= KERNEL_RADIUS_EXTENDED; ++j) {
            // Guard against out-of-bounds access
            int gx = global_x + j;
            int gy = global_y + i;
            if (!in_bounds(gx, gy, kernel_constants.width, kernel_constants.height)) {
                continue;
            }

            // Calculate the local coordinates
            int lx = local_x + j;
            int ly = local_y + i;

            pixel_t pixel = shared_image(lx, ly);
            uint8_t mask_pixel = shared_mask(lx, ly);
            uint8_t disp_mask_pixel = shared_dispersion_mask(lx, ly);
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
        bool disp_mask = shared_dispersion_mask(local_x, local_y) == MASKED_PIXEL;
        // The pixel must be above the global threshold
        bool global_mask = shared_image(local_x, local_y) > threshold;
        // Calculate the local mean
        float mean = (n > 1 ? sum_f / n : 0);  // If n is less than 1, set mean to 0
        // The pixel must be above the local threshold
        bool local_mask = shared_image(local_x, local_y)
                          >= (mean + kernel_constants.n_sig_s * sqrtf(mean));

        result_mask(global_x, global_y) = disp_mask && global_mask && local_mask;
    } else {
        result_mask(global_x, global_y) = 0;
    }
}
#pragma endregion