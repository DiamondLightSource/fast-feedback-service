#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "../spotfinder.cuh"
#include "erosion.cuh"

namespace cg = cooperative_groups;

#pragma region Device Functions
/**
 * @brief Load central pixels into shared memory.
 * @param block The cooperative group for the current block.
 * @param mask Pointer to the mask data.
 * @param shared_mask Pointer to the shared memory buffer.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius around each masked pixel to be considered.
 */
__device__ void load_central_pixels(cg::thread_block block,
                                    const uint8_t *mask,
                                    uint8_t *shared_mask,
                                    size_t mask_pitch,
                                    int width,
                                    int height,
                                    int radius) {
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;
    int local_x = block.thread_index().x + radius;
    int local_y = block.thread_index().y + radius;
    int shared_width = block.group_dim().x + 2 * radius;

    // Load central pixels into shared memory
    if (x < width && y < height) {
        shared_mask[local_y * shared_width + local_x] = mask[y * mask_pitch + x];
    } else {
        shared_mask[local_y * shared_width + local_x] = MASKED_PIXEL;
    }
}

/**
 * @brief Load border pixels into shared memory.
 * @param block The cooperative group for the current block.
 * @param mask Pointer to the mask data.
 * @param shared_mask Pointer to the shared memory buffer.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius around each masked pixel to be considered.
 */
__device__ void load_border_pixels(cg::thread_block block,
                                   const uint8_t *mask,
                                   uint8_t *shared_mask,
                                   size_t mask_pitch,
                                   int width,
                                   int height,
                                   int radius) {
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;
    int local_x = block.thread_index().x + radius;
    int local_y = block.thread_index().y + radius;
    int shared_width = block.group_dim().x + 2 * radius;
    int shared_height = block.group_dim().y + 2 * radius;

    // Load border pixels into shared memory
    for (int i = block.thread_index().x; i < shared_width; i += block.group_dim().x) {
        for (int j = block.thread_index().y; j < shared_height;
             j += block.group_dim().y) {
            int global_x = x + (i - local_x);
            int global_y = y + (j - local_y);

            bool is_within_central_region =
              (i >= radius && i < shared_width - radius && j >= radius
               && j < shared_height - radius);
            bool is_global_x_in_bounds = (global_x >= 0 && global_x < width);
            bool is_global_y_in_bounds = (global_y >= 0 && global_y < height);

            if (is_within_central_region) {
                continue;
            }

            if (is_global_x_in_bounds && is_global_y_in_bounds) {
                shared_mask[j * shared_width + i] =
                  mask[global_y * mask_pitch + global_x];
            } else {
                shared_mask[j * shared_width + i] = MASKED_PIXEL;
            }
        }
    }
}

/**
 * @brief Determine if the current pixel should be erased based on the mask.
 * @param block The cooperative group for the current block.
 * @param shared_mask Pointer to the shared memory buffer.
 * @param radius The radius around each masked pixel to be considered.
 * @param distance_threshold The maximum Chebyshev distance for erasing the current pixel.
 * @return True if the current pixel should be erased, false otherwise.
 */
__device__ bool determine_erasure(cg::thread_block block,
                                  const uint8_t *shared_mask,
                                  int radius,
                                  int distance_threshold) {
    int local_x = block.thread_index().x + radius;
    int local_y = block.thread_index().y + radius;
    int shared_width = block.group_dim().x + 2 * radius;

    bool should_erase = false;
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            if (shared_mask[(local_y + j) * shared_width + (local_x + i)]
                == MASKED_PIXEL) {
                int chebyshev_distance = max(abs(i), abs(j));
                if (chebyshev_distance <= distance_threshold) {
                    should_erase = true;
                    break;
                }
            }
        }
        if (should_erase) {
            break;
        }
    }
    return should_erase;
}

// __global__ void determine_erasure_kernel(const uint8_t *shared_mask,
//                                          int shared_width,
//                                          int local_x,
//                                          int local_y,
//                                          int radius,
//                                          int distance_threshold,
//                                          unsigned int *should_erase) {
//     int i = threadIdx.x - radius;
//     int j = threadIdx.y - radius;

//     if (shared_mask[(local_y + j) * shared_width + (local_x + i)] == MASKED_PIXEL) {
//         int chebyshev_distance = max(abs(i), abs(j));
//         if (chebyshev_distance <= distance_threshold) {
//             atomicExch(should_erase, 1u);
//         }
//     }
// }

/**
 * @brief Device function to determine if the current pixel should be erased using dynamic parallelism.
 * @param shared_mask Pointer to the shared memory buffer.
 * @param threadParams Thread-specific information for the current thread.
 * @param radius The radius around each masked pixel to be considered.
 * @param distance_threshold The maximum Chebyshev distance for erasing the current pixel.
 * @return True if the current pixel should be erased, false otherwise.
 */
// __device__ bool launch_determine_erasure_kernel(const uint8_t *shared_mask,
//                                                 const KernelThreadParams &threadParams,
//                                                 int radius,
//                                                 int distance_threshold) {
//     // Allocate memory for the erasure flag
//     unsigned int *d_should_erase;
//     cudaMalloc(&d_should_erase, sizeof(unsigned int));
//     cudaMemset(d_should_erase, 0, sizeof(unsigned int));

//     // Launch the erasure determination kernel
//     dim3 erasure_block_size(2 * radius + 1, 2 * radius + 1);
//     determine_erasure_kernel<<<1, erasure_block_size>>>(shared_mask,
//                                                         threadParams.shared_width,
//                                                         threadParams.local_x,
//                                                         threadParams.local_y,
//                                                         radius,
//                                                         distance_threshold,
//                                                         d_should_erase);

//     // Copy the result back to the host
//     unsigned int h_should_erase_uint;
//     cudaMemcpy(&h_should_erase_uint,
//                d_should_erase,
//                sizeof(unsigned int),
//                cudaMemcpyDeviceToHost);
//     cudaFree(d_should_erase);

//     return h_should_erase_uint == 1u;
// }
#pragma endregion Device Functions

#pragma region Erosion kernel(s)
/**
 * @brief CUDA kernel to apply erosion based on the mask and update the erosion_mask.
 * 
 * This kernel uses shared memory to store a local copy of the mask for each block.
 * 
 * @param mask Pointer to the mask data indicating valid pixels to be eroded.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius around each masked pixel to also be masked.
 */
__global__ void erosion_kernel(
  uint8_t __restrict__ *mask,
  // __restrict__ is a hint to the compiler that the two pointers are not
  // aliased, allowing the compiler to perform more agressive optimizations
  size_t mask_pitch,
  int width,
  int height,
  uint8_t radius) {
    // Declare shared memory to store a local copy of the mask for the block
    extern __shared__ uint8_t shared_mask[];

    // Create a cooperative group for the current block
    cg::thread_block block = cg::this_thread_block();

    // Load central pixels
    load_central_pixels(block, mask, shared_mask, mask_pitch, width, height, radius);

    // Load border pixels
    load_border_pixels(block, mask, shared_mask, mask_pitch, width, height, radius);

    // Synchronize threads to ensure all shared memory is loaded
    block.sync();

    /*
     * If the current pixel is outside the image bounds, return without doing anything.
     * We do this after loading shared memory as it may be necessary for this thread 
     * to load border pixels.
    */
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    // Return if the current pixel is outside the image bounds
    if (x >= width || y >= height) return;

    /*
     * If the current pixel is not a signal pixel, mark it as valid and return.
     * We do not need to perform erosion on non-signal pixels, but we need them
     * to be marked as valid in order to allow the background calculation to proceed.
    */
    if (mask[y * mask_pitch + x] == 0) {
        mask[y * mask_pitch + x] = VALID_PIXEL;
        return;
    }

    constexpr uint8_t chebyshev_distance_threshold = 2;

    // Determine if the current pixel should be erased
    bool should_erase =
      determine_erasure(block, shared_mask, radius, chebyshev_distance_threshold);
    // DIALS uses 2 as the Chebyshev distance threshold for erasing pixels

    // dynamic parrelism based
    // bool should_erase_gpu =
    //   launch_determine_erasure_kernel(shared_mask,
    //                                   threadParams,
    //                                   radius,
    //                                   2);  // Use 2 as the Chebyshev distance threshold

    // Update the erosion_mask based on erosion result
    if (should_erase) {
        /*
         * Erase the pixel from the background mask. This is done by setting the pixel
         * as valid (i.e. not masked) in the mask data. This allows the pixel to be
         * considered as a background pixel in the background calculation as it is not
         * considered part of the signal.
        */
        mask[y * mask_pitch + x] = VALID_PIXEL;
    } else {
        /*
         * If the pixel should not be erased, this means that it is part of the signal.
         * and needs to be marked as masked in the mask data. This prevents the pixel
         * from being considered as part of the background in the background calculation.
        */

        // Invert 'valid' signal spot to 'masked' background spots
        mask[y * mask_pitch + x] = !mask[y * mask_pitch + x];
    }
}
#pragma endregion Kernel