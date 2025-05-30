/**
 * @file erosion.cu
 * @brief Contains the CUDA kernel implementation for performing
 *        morphological erosion on an image using a given kernel radius
 *        and Chebyshev distance threshold.
 *
 * This kernel processes a dispersion mask containing potential signal
 * spots and background pixels. The kernel processes each pixel in the
 * mask and iterates over its local neighbourhood defined by a given
 * kernel radius. The kernel then checks if each pixel is within a given
 * Chebyshev distance threshold of a background pixel. If the pixel is
 * within the threshold, it is considered part of the background and is
 * marked as a background pixel in the erosion mask. Therefore eroding
 * the edges of the signal spots.
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
#include "erosion.cuh"

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

#pragma region Erosion kernel
/**
 * @brief CUDA kernel to perform morphological erosion on a dispersion mask
 *       using a given kernel radius and Chebyshev distance threshold.
 * 
 * @param dispersion_mask_ptr Pointer to the dispersion mask data
 * @param erosion_mask_ptr (Output) empty mask to store the erosion result
 * @param dispersion_mask_pitch Pitch of the dispersion mask data
 * @param erosion_mask_pitch Pitch of the erosion mask data
 * @param radius Radius of the erosion kernel
 */
__global__ void erosion(uint8_t __restrict__ *dispersion_mask_ptr,
                        uint8_t __restrict__ *erosion_mask_ptr,
                        uint8_t __restrict__ *mask_ptr,
                        size_t dispersion_mask_pitch,
                        size_t erosion_mask_pitch,
                        size_t mask_pitch,
                        uint8_t radius) {
    // Create pitched arrays for data access
    PitchedArray2D<uint8_t> dispersion_mask{dispersion_mask_ptr,
                                            &dispersion_mask_pitch};
    PitchedArray2D<uint8_t> erosion_mask{erosion_mask_ptr, &erosion_mask_pitch};
    PitchedArray2D<uint8_t> mask{mask_ptr, &mask_pitch};

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    // Guards
    if (x >= kernel_constants.width || y >= kernel_constants.height)
        return;  // Out of bounds guard

    bool is_background = dispersion_mask(x, y) == MASKED_PIXEL;

    if (is_background) {
        /*
         * If the pixel is masked, we want to set it to VALID_PIXEL
         * in order to invert the mask.
        */
        erosion_mask(x, y) = VALID_PIXEL;
        return;
    }

    bool should_erase = false;  // Flag to determine if the pixel should be erased
    constexpr uint8_t chebyshev_distance_threshold = 2;

    // Iterate over the kernel bounds
    for (int i = -radius; i <= radius; ++i) {      // Iterate over y offsets
        for (int j = -radius; j <= radius; ++j) {  // Iterate over x offsets
            int lx = x + j;                        // Offset x coordinate
            int ly = y + i;                        // Offset y coordinate

            // Check if the pixel is masked
            if (mask(lx, ly) == 0) {
                // If so, skip the pixel
                continue;
            }

            // Get pixel from step in kernel
            uint8_t this_pixel = dispersion_mask(lx, ly);

            if (this_pixel == MASKED_PIXEL) {
                // If the current pixel is background, check the Chebyshev distance
                uint8_t chebyshev_distance = max(abs(i), abs(j));

                if (chebyshev_distance <= chebyshev_distance_threshold) {
                    // If a background pixel is too close, the current pixel should be erased
                    should_erase = true;
                    // We can then break out of the loop, as no further checks are necessary
                    goto termination;
                }
            }
        }
    }

termination:
    if (should_erase) {
        /*
         * Erase the pixel from the background mask. This is done by setting the pixel
         * as valid (i.e. not masked) in the erosion_mask data. This allows the pixel to be
         * considered as a background pixel in the background calculation as it is not
         * considered part of the signal.
        */
        erosion_mask(x, y) = VALID_PIXEL;
    } else {
        /*
         * If the pixel should not be erased, this means that it is part of the signal.
         * and needs to be marked as masked in the erosion_mask data. This prevents the pixel
         * from being considered as part of the background in the background calculation.
        */

        // Invert 'valid' signal spot to 'masked' background spots
        erosion_mask(x, y) = !dispersion_mask(x, y);
    }
}
#pragma enregion Erosion kernel