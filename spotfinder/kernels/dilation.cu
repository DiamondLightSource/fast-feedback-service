#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "../spotfinder.cuh"
#include "dilation.cuh"

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
 * @brief Determine if the current pixel should be included based on the surrounding pixels.
 * @param block The cooperative group for the current block.
 * @param shared_mask Pointer to the shared memory buffer.
 * @param radius The radius around each masked pixel to be considered.
 * @param distance_threshold The maximum Chebyshev distance for including the current pixel.
 * @return True if the current pixel should be included, false otherwise.
 */
__device__ bool determine_inclusion(cg::thread_block block,
                                    const uint8_t *shared_mask,
                                    int radius,
                                    int distance_threshold) {
    int local_x = block.thread_index().x + radius;
    int local_y = block.thread_index().y + radius;
    int shared_width = block.group_dim().x + 2 * radius;

    bool should_include = false;
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            if (shared_mask[(local_y + j) * shared_width + (local_x + i)]
                == VALID_PIXEL) {
                int chebyshev_distance = max(abs(i), abs(j));
                if (chebyshev_distance <= distance_threshold) {
                    should_include = true;
                    break;
                }
            }
        }
        if (should_include) {
            break;
        }
    }
    return should_include;
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

#pragma region Kernel
__global__ void dilation_kernel(uint8_t __restrict__ *mask,
                                size_t mask_pitch,
                                int width,
                                int height,
                                int radius) {
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

    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    // Return if the current pixel is outside the image bounds
    if (x >= width || y >= height) return;

    /*
     * If the current pixel is a signal pixel, return as we do not want to change it.
     */
    if (mask[y * mask_pitch + x] == VALID_PIXEL) return;

    // Determine if the current pixel should be included in the signal
    bool should_include = determine_inclusion(block, shared_mask, radius, 2);

    // Update the mask based on the dilation result
    if (should_include) {
        mask[y * mask_pitch + x] = VALID_PIXEL;
    }
}
#pragma endregion Kernel