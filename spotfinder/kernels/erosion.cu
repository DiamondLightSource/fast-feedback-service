#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "../spotfinder.cuh"
#include "erosion.cuh"

// Macro to get the value of a pitched array
#define GET_PITCHED_VALUE(array, pitch, x, y) (array[y * pitch + x])

namespace cg = cooperative_groups;

#pragma region Erosion kernel
__global__ void erosion_kernel(
  uint8_t __restrict__ *dispersion_mask,
  uint8_t __restrict__ *erosion_mask,
  uint8_t __restrict__ *mask,
  // __restrict__ is a hint to the compiler that the two pointers are not
  // aliased, allowing the compiler to perform more agressive optimizations
  size_t dispersion_mask_pitch,
  size_t erosion_mask_pitch,
  size_t mask_pitch,
  int width,
  int height,
  uint8_t radius) {
    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    // Guards
    if (x >= width || y >= height) return;  // Out of bounds guard

    bool is_background = dispersion_mask[y * dispersion_mask_pitch + x] == MASKED_PIXEL;
    if (is_background) {
        /*
         * If the pixel is masked, we want to set it to VALID_PIXEL
         * in order to invert the mask.
        */
        erosion_mask[y * erosion_mask_pitch + x] = VALID_PIXEL;
        return;
    }

    // Calculate the bounds of the erosion kernel
    int x_start = max(0, x - radius);
    int x_end = min(x + radius + 1, width);
    int y_start = max(0, y - radius);
    int y_end = min(y + radius + 1, height);

    bool should_erase = false;  // Flag to determine if the pixel should be erased
    constexpr uint8_t chebyshev_distance_threshold = 2;

    // Iterate over the kernel bounds
    for (int kernel_x = x_start; kernel_x < x_end; ++kernel_x) {
        for (int kernel_y = y_start; kernel_y < y_end; ++kernel_y) {
            /*
             * TODO: Investigate whether we should be doing this or not!
             * Intuition says that we should be considering the mask,
             * however DIALS does not do this. May be a bug, may be on
             * purpose? Investigate!
            */
            // if (mask[kernel_y * mask_pitch + kernel_x] == 0) {
            //     continue;
            // }
            if (dispersion_mask[kernel_y * dispersion_mask_pitch + kernel_x]
                == MASKED_PIXEL) {
                // If the current pixel is background, check the Chebyshev distance
                uint8_t chebyshev_distance = max(abs(kernel_x - x), abs(kernel_y - y));

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
        erosion_mask[y * erosion_mask_pitch + x] = VALID_PIXEL;
    } else {
        /*
         * If the pixel should not be erased, this means that it is part of the signal.
         * and needs to be marked as masked in the erosion_mask data. This prevents the pixel
         * from being considered as part of the background in the background calculation.
        */

        // Invert 'valid' signal spot to 'masked' background spots
        erosion_mask[y * erosion_mask_pitch + x] =
          !dispersion_mask[y * dispersion_mask_pitch + x];
    }
}
#pragma enregion Erosion kernel