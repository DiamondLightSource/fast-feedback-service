#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <type_traits>

#include "cuda_common.hpp"
#include "device_common.cuh"
#include "ffs_logger.hpp"
#include "h5read.h"
#include "integrator.cuh"
//#include "integrator/background.cuh"
#include "integrator/extent.hpp"
#include "ellipsoid.cuh"
#include "math/math_utils.cuh"

__global__ void ellipsoid_integrate(pixel_t *d_image,
                        size_t image_pitch,
                        uint32_t width,
                        uint32_t height,
                        int image_num,
                        const BoundingBoxExtents *d_bboxes,
                        const size_t *d_reflection_indices,
                        size_t num_reflections_this_image,
                        const uint8_t *d_mask,
                        uint32_t max_corner_tile_corners,
                        accumulator_t *d_foreground_sum,
                        uint32_t *d_foreground_count,
                        uint8_t *d_success){
    // One block per reflection shoebox on this image; blockIdx.x indexes the
    // image-level reflection list directly.
    const size_t block_refl = blockIdx.x;
    if (block_refl >= num_reflections_this_image) return;
    const size_t refl_idx = d_reflection_indices[block_refl];
    const BoundingBoxExtents bbox = d_bboxes[refl_idx];
    
    // Add assert once sorted out updating bbox.
    // assert(image_num >= bbox.z_min && image_num < bbox.z_max);

    // Pitched image accessor
    size_t elem_pitch = image_pitch / sizeof(pixel_t);
    PitchedArray2D<pixel_t> image(d_image, &elem_pitch);
    const int w = bbox.x_max - bbox.x_min;
    const int h = bbox.y_max - bbox.y_min;
    const int npix = w * h;

    // Classify + accumulate one shoebox pixel. Shared by the tile and fallback
    // paths; the accumulation itself is unchanged from the per-pixel kernel.
    auto accumulate = [&](int px, int py, bool any_fg) {
        // Out-of-image pixels still classify (geometry is valid beyond the
        // edge); a foreground pixel outside the image fails the reflection.
        const bool pixel_in_image = (px >= 0) && (px < static_cast<int>(width))
                                    && (py >= 0) && (py < static_cast<int>(height));

        // Both paths below need the mask flag and the pixel value, so the two
        // loads are issued together rather than letting the mask test gate the
        // image load and serialise two L1TEX round trips per pixel. Hoisting
        // them past the corner-fill barrier goes further still, but costs
        // registers, and registers bind occupancy on this kernel.
        uint8_t mask_flag = 0;
        pixel_t raw_value = 0;
        if (pixel_in_image) {
            mask_flag = d_mask[static_cast<size_t>(py) * width + px];
            raw_value = image(px, py);
        }

        if (any_fg) {
            // Foreground outside the image or on a masked pixel makes the
            // reflection unusable. d_success is only ever cleared, never set,
            // so the concurrent writes need no atomic.
            if (!pixel_in_image || mask_flag == 0) {
                d_success[refl_idx] = 0;
            } else {
                const accumulator_t pixel_value = static_cast<accumulator_t>(raw_value);
                atomicAdd(&d_foreground_sum[refl_idx], pixel_value);
                atomicAdd(&d_foreground_count[refl_idx], 1u);
            }
        }
    };

    for (int p = static_cast<int>(threadIdx.x); p < npix;
             p += static_cast<int>(blockDim.x)) {
            const int px = bbox.x_min + (p % w);
            const int py = bbox.y_min + (p / w);
            accumulate(px, py, true);
             }
}

void compute_ellipsoid_integrate(pixel_t *d_image,
                        size_t image_pitch,
                        uint32_t width,
                        uint32_t height,
                        int image_num,
                        const BoundingBoxExtents *d_bboxes,
                        const size_t *d_reflection_indices,
                        size_t num_reflections_this_image,
                        const uint8_t *d_mask,
                        uint32_t max_corner_tile_corners,
                        accumulator_t *d_foreground_sum,
                        uint32_t *d_foreground_count,
                        uint8_t *d_success,
                    cudaStream_t stream){
    if (num_reflections_this_image == 0) return;

    // Configure kernel launch parameters.
    dim3 blockDim(ELLIPSOID_THREADS);
    dim3 gridDim(static_cast<uint32_t>(num_reflections_this_image));

    static const uint32_t max_smem_per_block = [] {
        int device = 0;
        cudaGetDevice(&device);
        int bytes = 0;
        cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, device);
        return static_cast<uint32_t>(bytes);
    }();
    const uint32_t tile_capacity =
      std::min(max_corner_tile_corners, max_smem_per_block);
    const size_t smem_bytes = tile_capacity * sizeof(uint8_t);
    static std::atomic<uint32_t> smem_high_water{0};
    uint32_t prev_high = smem_high_water.load(std::memory_order_relaxed);
    if (smem_bytes > prev_high
        && smem_high_water.compare_exchange_strong(
            prev_high, static_cast<uint32_t>(smem_bytes))) {
        int blocks_no_smem = 0;
        int blocks_with_smem = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_no_smem,
            ellipsoid_integrate,
            static_cast<int>(ELLIPSOID_THREADS),
            0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_with_smem,
            ellipsoid_integrate,
            static_cast<int>(ELLIPSOID_THREADS),
            smem_bytes);
        if (blocks_with_smem < blocks_no_smem) {
            static std::once_flag occupancy_warning_flag;
            std::call_once(occupancy_warning_flag, [&] {
                logger.warn(
                    "Kabsch corner tile of {} B (largest shoebox on image {}) "
                    "reduces occupancy from {} to {} blocks/SM; further "
                    "occurrences are not reported",
                    smem_bytes,
                    image_num,
                    blocks_no_smem,
                    blocks_with_smem);
            });
        }
    }
    ellipsoid_integrate<<<gridDim, blockDim, smem_bytes, stream>>>(
        d_image,
        image_pitch,
        width,
        height,
        image_num,
        d_bboxes,
        d_reflection_indices,
        num_reflections_this_image,
        d_mask,
        max_corner_tile_corners,
        d_foreground_sum,
        d_foreground_count,
        d_success);
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
          fmt::format("Kernel launch failed: {}", cudaGetErrorString(err)));
    }
}