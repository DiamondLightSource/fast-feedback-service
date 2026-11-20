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
#include "integrator/background.cuh"
#include "integrator/extent.hpp"
#include "ellipsoid.cuh"
#include "kabsch.cuh"
#include "math/math_utils.cuh"

using namespace fastvec;

constexpr scalar_t ellipsoid_quantile_D = 11.8292; //double D = chisq_quantile(2, probability_);


__device__ __forceinline__ bool is_foreground_ellipsoid(
    const Vec2D& eps,
    const Vec2D& mubar,
    const Mat2D& Sbar_inv, // 4 values in order (2x2 matrix)
    const scalar_t wavelength) {
    // Calculation is d = AT_B_A(x1 - mubar, Sbar_inv);
    scalar_t x0 = eps.x / wavelength;
    scalar_t x1 = eps.y / wavelength;
    scalar_t d0 = x0 - mubar.x;
    scalar_t d1 = x1 - mubar.y;
    scalar_t d =
       d0 * d0 * Sbar_inv.xx +
       d0 * d1 * Sbar_inv.xy +
       d0 * d1 * Sbar_inv.yx +
       d1 * d1 * Sbar_inv.yy;                                         
 
    return d < ellipsoid_quantile_D;
}


__device__ Vec2D pixel_to_kabsch_ellipsoid(const Vector3D &s0,
                                    const Vector3D &s1_c,
                                    const Vector3D &s_pixel) {
    Vector3D e1 = normalized(cross(s1_c, s0));
    Vector3D e2 = normalized(cross(s1_c, e1));
    scalar_t s1_len = norm(s1_c);
    Vector3D deltaS = s_pixel - s1_c;
    scalar_t eps1 = dot(e1, deltaS) / s1_len;
    scalar_t eps2 = dot(e2, deltaS) / s1_len;  
    return {eps1, eps2};                            
}


__device__ __forceinline__ bool corner_is_foreground_ellipsoid(
  int cx,
  int cy,
  const Vector3D &s0,
  const Vector3D &s2c,
  const Vec2D &mubar,
  const scalar_t *d_matrix,
  const Mat2D& Sbar_inv,
  scalar_t wavelength,
  const DetectorParameters &det_params) {
    const Vector3D s_pixel = corner_s_pixel(cx, cy, d_matrix, wavelength, det_params);
    const Vec2D eps = pixel_to_kabsch_ellipsoid(s0, s2c, s_pixel);
    return is_foreground_ellipsoid(eps, mubar, Sbar_inv, wavelength);
}


__device__ __forceinline__ bool pixel_is_foreground_ellipsoid(
  int px,
  int py,
  const Vector3D &s0,
  const Vector3D &s2c,
  const Vec2D& mubar,
  const scalar_t *d_matrix,
  const Mat2D& Sbar_inv,
  scalar_t wavelength,
  const DetectorParameters &det_params) {
#pragma unroll
    for (int dy = 0; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = 0; dx <= 1; ++dx) {
            if (corner_is_foreground_ellipsoid(px + dx,
                                    py + dy,
                                           s0,
                                           s2c,
                                           mubar,
                                           d_matrix,
                                           Sbar_inv,
                                           wavelength,
                                           det_params)) {
                return true;
            }
        }
    }
    return false;
}

__global__ void ellipsoid_integrate(pixel_t *d_image,
                        size_t image_pitch,
                        uint32_t width,
                        uint32_t height,
                        int image_num,
                        const scalar_t *d_matrix,
                        const scalar_t wavelength,
                        const DetectorParameters det_params,
                        const Vector3D s0,
                        const BoundingBoxExtents *d_bboxes,
                        const size_t *d_reflection_indices,
                        const Vector3D *d_s2_vectors,
                        const Vec2D *d_mubar,
                        const Mat2D *d_sbar_inv,
                        size_t num_reflections_this_image,
                        const uint8_t *d_mask,
                        uint32_t tile_capacity,
                        accumulator_t *d_foreground_sum,
                        uint32_t *d_foreground_count,
                        BackgroundPool background_pool,
                        uint32_t *d_background_spill,
                        unsigned long long *d_intensity_times_x,
                        unsigned long long *d_intensity_times_y,
                        uint8_t *d_success){
    // One block per reflection shoebox on this image; blockIdx.x indexes the
    // image-level reflection list directly.
    const size_t block_refl = blockIdx.x;
    if (block_refl >= num_reflections_this_image) return;
    const size_t refl_idx = d_reflection_indices[block_refl];
    const Vector3D s2c = d_s2_vectors[refl_idx];
    const BoundingBoxExtents bbox = d_bboxes[refl_idx];
    const Vec2D mubar = d_mubar[refl_idx];
    const Mat2D Sbar_inv = d_sbar_inv[refl_idx];
    
    // Pitched image accessor
    size_t elem_pitch = image_pitch / sizeof(pixel_t);
    PitchedArray2D<pixel_t> image(d_image, &elem_pitch);
    const int w = bbox.x_max - bbox.x_min;
    const int h = bbox.y_max - bbox.y_min;
    const int npix = w * h;

    extern __shared__ uint8_t s_corner[];

    auto corner_fg = [&](int cx, int cy) -> bool {
        return corner_is_foreground_ellipsoid(
            cx, cy, s0, s2c, mubar, d_matrix, Sbar_inv, wavelength, det_params
        );
    };

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
                // Accumulate intensity·coord for centre-of-mass (xyzobs.px).
                // Baseline uses (x+0.5, y+0.5, 0.5); we approximate the
                // half-pixel offset by multiplying by 2x+1 etc. and dividing
                // by 2 in finalisation to stay in integer atomics.
                const unsigned long long pv64 =
                  static_cast<unsigned long long>(pixel_value);
                atomicAdd(&d_intensity_times_x[refl_idx],
                          pv64 * static_cast<unsigned long long>(2 * px + 1));
                atomicAdd(&d_intensity_times_y[refl_idx],
                          pv64 * static_cast<unsigned long long>(2 * py + 1));
            }
        } else {
            if (pixel_in_image && mask_flag != 0) {
                const int value = static_cast<int>(raw_value);
                if (value >= 0) {
                    background_accumulate(background_pool,
                                          static_cast<uint32_t>(refl_idx),
                                          &d_background_spill[refl_idx],
                                          static_cast<uint32_t>(value));
                }
            }
        }
    };

    const int cw = w + 1;
    const int ch = h + 1;

    if (static_cast<uint32_t>(cw) * static_cast<uint32_t>(ch) <= tile_capacity) {
        // Fill the corner tile: one corner_is_foreground per corner, threads
        // striding densely over the (w+1)×(h+1) grid.
        const int num_corners = cw * ch;
        for (int c = static_cast<int>(threadIdx.x); c < num_corners;
             c += static_cast<int>(blockDim.x)) {
            const int lcx = c % cw;
            const int lcy = c / cw;
            s_corner[c] = corner_fg(bbox.x_min + lcx, bbox.y_min + lcy) ? 1u : 0u;
        }
        __syncthreads();

        // Read the tile: each pixel ORs its four corner bits.
        for (int p = static_cast<int>(threadIdx.x); p < npix;
             p += static_cast<int>(blockDim.x)) {
            const int lx = p % w;
            const int ly = p / w;
            const bool any_fg = s_corner[ly * cw + lx] | s_corner[ly * cw + lx + 1]
                                | s_corner[(ly + 1) * cw + lx]
                                | s_corner[(ly + 1) * cw + lx + 1];
            accumulate(bbox.x_min + lx, bbox.y_min + ly, any_fg);
        }
    } else {
        for (int p = static_cast<int>(threadIdx.x); p < npix;
            p += static_cast<int>(blockDim.x)) {
            const int px = bbox.x_min + (p % w);
            const int py = bbox.y_min + (p / w);
            accumulate(
                px,
                py,
                pixel_is_foreground_ellipsoid(px, py, s0, s2c,
                    mubar,
                    d_matrix,
                    Sbar_inv,
                    wavelength,
                    det_params)
            );
        }
    }
}

void compute_ellipsoid_integrate(pixel_t *d_image,
                        size_t image_pitch,
                        uint32_t width,
                        uint32_t height,
                        int image_num,
                        const scalar_t *d_matrix,
                        const scalar_t wavelength,
                        const DetectorParameters det_params,
                        const Vector3D s0,
                        const BoundingBoxExtents *d_bboxes,
                        const size_t *d_reflection_indices,
                        const Vector3D *d_s2_vectors,
                        const Vec2D *d_mubar,
                        const Mat2D *d_sbar_inv,
                        size_t num_reflections_this_image,
                        const uint8_t *d_mask,
                        uint32_t max_corner_tile_corners,
                        accumulator_t *d_foreground_sum,
                        uint32_t *d_foreground_count,
                        BackgroundPool background_pool,
                        uint32_t *d_background_spill,
                        unsigned long long *d_intensity_times_x,
                        unsigned long long *d_intensity_times_y,
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
        d_matrix,
        wavelength,
        det_params,
        s0,
        d_bboxes,
        d_reflection_indices,
        d_s2_vectors,
        d_mubar,
        d_sbar_inv,
        num_reflections_this_image,
        d_mask,
        tile_capacity,
        d_foreground_sum,
        d_foreground_count,
        background_pool,
        d_background_spill,
        d_intensity_times_x,
        d_intensity_times_y,
        d_success);
    // Check for kernel launch errors
    cuda_check_launch("ellipsoid_transform");
}