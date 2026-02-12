/**
 * @file kabsch.cu
 * @brief CUDA implementation of Kabsch coordinate transformations for
 * integration
 * 
 * This file implements GPU-accelerated coordinate transformations used
 * in the Kabsch algorithm for data processing. The
 * Kabsch transformation converts pixel coordinates from reciprocal
 * space into a geometry-invariant local coordinate system, enabling
 * efficient reflection profile integration and summation.
 * 
 * The coordinate system is defined by three basis vectors:
 * - e₁: Perpendicular to the scattering plane (s₁ᶜ × s₀)
 * - e₂: Within the scattering plane, orthogonal to e₁ (s₁ᶜ × e₁)  
 * - e₃: Bisects the incident and diffracted beam directions (s₁ᶜ + s₀)
 * 
 * The resulting Kabsch coordinates (ε₁, ε₂, ε₃) represent:
 * - ε₁: Displacement perpendicular to the scattering plane
 * - ε₂: Displacement within the scattering plane (with rotation correction)
 * - ε₃: Displacement along the rotation axis (scaled by geometry factor ζ)
 * 
 * This implementation uses modern C++ RAII principles with DeviceBuffer
 * for automatic GPU memory management and the custom Vector3D class for
 * mathematical operations. All vector operations are optimized for both
 * host and device execution contexts.
 *
 * @author Dimitrios Vlachos
 * @date 2025
 * @see Vector3D for mathematical vector operations
 * @see DeviceBuffer for GPU memory management
 * 
 * References:
 * - Kabsch, W. (2010). Acta Cryst. D66, 133–144
 *
 * GOAL: Determine foreground/background pixels and atomically aggregate
 * intensities per reflection within the kernel, avoiding the need to return
 * all epsilon values (too much data).
 */

#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_common.hpp"
#include "device_common.cuh"
#include "extent.hpp"
#include "h5read.h"
#include "integrator.cuh"
#include "kabsch.cuh"
#include "math/math_utils.cuh"

using namespace fastvec;

// Block dimensions for the Kabsch transform kernel.
// Used by both the kernel (shared memory sizing) and host wrapper (grid launch).
//
// Each block of KABSCH_BLOCK_X × KABSCH_BLOCK_Y threads maps 1:1 to corners.
// A pixel's voxel has 4 xy-corners: (px,py), (px+1,py), (px,py+1), (px+1,py+1).
// All 4 must be in the same block's shared memory so the pixel thread can read
// them after __syncthreads(). This means a block of 32×16 corners can only
// service (32-1)×(16-1) = 31×15 complete pixels — the rightmost column and
// bottom row of threads are "helper" corners that exist solely so the last
// interior pixel has a right/bottom neighbour to read.
//
// The grid is launched with a stride of (BLOCK-1) so adjacent blocks overlap
// by one column/row of corners.  That single shared column/row is computed
// by both blocks (redundantly), which costs ~3-6% extra work but keeps
// everything in fast shared memory within a single kernel launch.
constexpr int KABSCH_BLOCK_X = 32;
constexpr int KABSCH_BLOCK_Y = 16;

// Maximum number of reflections that can overlap a single block's footprint
constexpr size_t MAX_BLOCK_REFLECTIONS = 64;

/**
 * @brief Check if a pixel is in the foreground region of a reflection
 *
 * Uses the ellipsoid condition in Kabsch space to classify pixels:
 *   ε₁²/δ_B² + ε₂²/δ_B² + ε₃²/δ_M² ≤ 1.0
 *
 * @param eps Kabsch coordinates (ε₁, ε₂, ε₃)
 * @param delta_b Foreground extent in e₁/e₂ directions (n_sigma × σ_D)
 * @param delta_m Foreground extent in e₃ direction (n_sigma × σ_M)
 * @return true if pixel is foreground, false if background
 */
__device__ __forceinline__ bool is_foreground(const Vector3D &eps,
                                              scalar_t delta_b,
                                              scalar_t delta_m) {
    // Precompute inverse squares to avoid repeated division
    // Division is more expensive than multiplication on GPU
    // -> Verify in profiler how compiler optimises this PTX
    scalar_t inv_delta_b_sq = scalar_t(1.0) / (delta_b * delta_b);
    scalar_t inv_delta_m_sq = scalar_t(1.0) / (delta_m * delta_m);

    // Ellipsoid condition: ε₁²/δ_B² + ε₂²/δ_B² + ε₃²/δ_M² ≤ 1.0
    scalar_t ellipsoid_val = (eps.x * eps.x + eps.y * eps.y) * inv_delta_b_sq
                             + (eps.z * eps.z) * inv_delta_m_sq;

    return ellipsoid_val <= scalar_t(1.0);
}

/**
 * @brief Transform a pixel from reciprocal space into the local Kabsch
   coordinate frame
 * 
 * Given a predicted reflection centre and a pixel's position in
 * reciprocal space, this function calculates the local Kabsch
 * coordinates (ε₁, ε₂, ε₃), which represent displacements along a
 * non-orthonormal basis defined by the scattering geometry.
 *
 * This is used to determine whether a pixel falls within the profile of
 * a reflection in Kabsch space, which allows summation or profile
 * integration to proceed in a geometry-invariant coordinate frame.
 *
 * @param s0 Incident beam vector (s₀), units of 1/Å
 * @param s1_c Predicted diffracted vector at the reflection centre (s₁ᶜ), units of 1/Å
 * @param phi_c Rotation angle at the reflection centre (φᶜ), in radians
 * @param s_pixel Diffracted vector at the current pixel (S′), units of 1/Å
 * @param phi_pixel Rotation angle at the pixel (φ′), in radians
 * @param rot_axis Unit goniometer rotation axis vector (m₂)
 * @param s1_len_out Output parameter for magnitude of s₁ᶜ (|s₁|)
 * @return Transformed coordinates in Kabsch space (ε₁, ε₂, ε₃)
 */
__device__ Vector3D pixel_to_kabsch(const Vector3D &s0,
                                    const Vector3D &s1_c,
                                    scalar_t phi_c,
                                    const Vector3D &s_pixel,
                                    scalar_t phi_pixel,
                                    const Vector3D &rot_axis,
                                    scalar_t &s1_len_out) {
    // Define the local Kabsch basis vectors:

    // e1 is perpendicular to the scattering plane
    Vector3D e1 = normalized(cross(s1_c, s0));

    // e2 lies within the scattering plane, orthogonal to e1
    Vector3D e2 = normalized(cross(s1_c, e1));

    // e3 bisects the angle between s0 and s1
    Vector3D e3 = normalized(s1_c + s0);

    // Compute the length of the predicted diffracted vector (|s₁|)
    scalar_t s1_len = norm(s1_c);
    s1_len_out = s1_len;

    // Rotation offset between the pixel and reflection centre
    scalar_t dphi = phi_pixel - phi_c;

    // Compute the predicted diffracted vector at φ′
    Vector3D s1_phi_prime = s1_c + e3 * dphi;

    // Difference vector between pixel's s′ and the φ′-adjusted centroid
    Vector3D deltaS = s_pixel - s1_phi_prime;

    // ε₁: displacement along e1, normalised by |s₁|
    scalar_t eps1 = dot(e1, deltaS) / s1_len;

    // ε₂: displacement along e2, normalised by |s₁|
    scalar_t eps2 = dot(e2, deltaS) / s1_len;

    // ε₃: displacement along rotation axis, scaled by ζ = m₂ · e₁
    scalar_t zeta = dot(rot_axis, e1);
    scalar_t eps3 = zeta * dphi;

    return make_vector3d(eps1, eps2, eps3);
}

/**
 * @brief CUDA kernel with 8-corner voxel foreground classification
 *
 * Each thread represents a corner in the (width+1)×(height+1) corner grid.
 * Blocks use overlapping tiles: KABSCH_BLOCK_X × KABSCH_BLOCK_Y threads
 * cover that many corners, yielding (KABSCH_BLOCK_X-1) × (KABSCH_BLOCK_Y-1)
 * complete pixels per block.  Adjacent blocks share one row/column of corners.
 *
 * Algorithm:
 *   1. Every thread computes Kabsch coordinates for its corner at two φ values
 *      (lower and upper z-faces of the voxel) and writes foreground flags to
 *      shared memory.
 *   2. After sync, interior threads (representing complete pixels) check all
 *      8 voxel corners. If ANY corner is foreground, the pixel is foreground;
 *      otherwise it's background.
 *
 * @param d_image Pointer to image data
 * @param image_pitch Pitch of the image data in bytes
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param image_num Current image number in the sequence
 * @param d_matrix Detector coordinate transformation matrix (3x3)
 * @param wavelength X-ray wavelength in Angstroms
 * @param osc_start Starting oscillation angle in degrees
 * @param osc_width Oscillation width per image in degrees
 * @param image_range_start First image number in the range
 * @param s0 Incident beam vector
 * @param rot_axis Rotation axis vector
 * @param d_s1_vectors Array of s1 vectors for each reflection
 * @param d_phi_values Array of phi values for each reflection
 * @param d_bboxes Array of bounding box structs for each reflection
 * @param d_reflection_indices Indices of reflections touching this image
 * @param num_reflections_this_image Number of reflections to process
 * @param delta_b Foreground extent in e₁/e₂ directions (n_sigma × σ_D)
 * @param delta_m Foreground extent in e₃ direction (n_sigma × σ_M)
 * @param d_foreground_sum Output: accumulated foreground intensities per reflection
 * @param d_foreground_count Output: foreground pixel counts per reflection
 * @param d_background_sum Output: accumulated background intensities per reflection
 * @param d_background_count Output: background pixel counts per reflection
 */
__global__ void kabsch_transform(pixel_t *d_image,
                                 size_t image_pitch,
                                 uint32_t width,
                                 uint32_t height,
                                 int image_num,
                                 const scalar_t *d_matrix,
                                 scalar_t wavelength,
                                 scalar_t osc_start,
                                 scalar_t osc_width,
                                 int image_range_start,
                                 Vector3D s0,
                                 Vector3D rot_axis,
                                 const Vector3D *d_s1_vectors,
                                 const scalar_t *d_phi_values,
                                 const BoundingBoxExtents *d_bboxes,
                                 const size_t *d_reflection_indices,
                                 size_t num_reflections_this_image,
                                 // Summation integration parameters
                                 scalar_t delta_b,
                                 scalar_t delta_m,
                                 // Output accumulators (atomically updated)
                                 accumulator_t *d_foreground_sum,
                                 uint32_t *d_foreground_count,
                                 accumulator_t *d_background_sum,
                                 uint32_t *d_background_count) {
    // ── Shared Memory ──────────────────────────────────────────────
    // Foreground status for each corner at two phi values (lower/upper z-face)
    __shared__ bool s_fg[2][KABSCH_BLOCK_Y][KABSCH_BLOCK_X];

    // Block-level reflection list: indices into the global reflection arrays
    __shared__ size_t s_block_refl[MAX_BLOCK_REFLECTIONS];
    __shared__ size_t s_num_block_refl;

    const int tx = threadIdx.x;  // Thread-local x index (also indexes shared memory)
    const int ty = threadIdx.y;  // Thread-local y index

    // Global corner coordinates.
    // The stride between block origins is (BLOCK-1), NOT BLOCK.  Compare:
    //   Normal:  gx = blockIdx.x * blockDim.x + tx          (stride 32)
    //   Here:    gx = blockIdx.x * (KABSCH_BLOCK_X - 1) + tx (stride 31)
    //
    // This means Block 0 covers corners 0-31, Block 1 covers 31-62, etc.
    // Corner 31 is computed by BOTH blocks — that's the overlap.  It ensures
    // pixel 30 (in Block 0) can read its right-side corner (31) from shared
    // memory, and pixel 31 (in Block 1) can read its left-side corner (31)
    // from its own block's shared memory.

    const int gx = blockIdx.x * (KABSCH_BLOCK_X - 1) + tx;  // Global x coordinate
    const int gy = blockIdx.y * (KABSCH_BLOCK_Y - 1) + ty;  // Global y coordinate

    // ── Per-Corner Setup ────────────────────────────────────────────
    // Two phi values define the z-extent of the voxel on this image
    const scalar_t phi_low = degrees_to_radians(
      osc_start + (static_cast<scalar_t>(image_num - image_range_start) * osc_width));
    const scalar_t phi_high = degrees_to_radians(
      osc_start
      + (static_cast<scalar_t>(image_num - image_range_start + 1) * osc_width));

    // s_pixel depends only on detector position (gx, gy), not phi
    const bool corner_valid = (gx >= 0) && (gx <= static_cast<int>(width)) && (gy >= 0)
                              && (gy <= static_cast<int>(height));

    Vector3D s_pixel = make_vector3d(0, 0, 0);
    if (corner_valid) {
        Vector3D lab_coord =
          make_vector3d(d_matrix[0] * gx + d_matrix[1] * gy + d_matrix[2],
                        d_matrix[3] * gx + d_matrix[4] * gy + d_matrix[5],
                        d_matrix[6] * gx + d_matrix[7] * gy + d_matrix[8]);
        s_pixel = normalized(lab_coord) / wavelength;
    }

    // ── Block Reflection Filter ─────────────────────────────────────
    // Thread 0 scans the image-level reflection list and keeps only those
    // whose bbox overlaps this block's corner footprint.
    if (tx == 0 && ty == 0) {
        s_num_block_refl = 0;

        const int bx_min = static_cast<int>(blockIdx.x) * (KABSCH_BLOCK_X - 1);
        const int bx_max = bx_min + KABSCH_BLOCK_X - 1;
        const int by_min = static_cast<int>(blockIdx.y) * (KABSCH_BLOCK_Y - 1);
        const int by_max = by_min + KABSCH_BLOCK_Y - 1;

        for (size_t i = 0; i < num_reflections_this_image; ++i) {
            const size_t refl_idx = d_reflection_indices[i];
            const BoundingBoxExtents &bbox = d_bboxes[refl_idx];

            // Does the reflection's bbox overlap this block AND current image?
            const bool overlaps =
              (bx_max >= bbox.x_min && bx_min <= bbox.x_max)
              && (by_max >= bbox.y_min && by_min <= bbox.y_max)
              && (image_num >= bbox.z_min && image_num < bbox.z_max);

            if (overlaps && s_num_block_refl < MAX_BLOCK_REFLECTIONS) {
                s_block_refl[s_num_block_refl++] = refl_idx;
            }
        }
    }

    __syncthreads();

    // Early exit if no reflections overlap this block
    if (s_num_block_refl == 0) return;
    // Pitched image accessor
    size_t elem_pitch = image_pitch / sizeof(pixel_t);
    PitchedArray2D<pixel_t> image(d_image, &elem_pitch);

    // Does this thread represent a complete pixel?
    // A pixel at (tx, ty) reads corners (tx, ty), (tx+1, ty), (tx, ty+1),
    // (tx+1, ty+1) from shared memory.  For tx+1 and ty+1 to be valid
    // indices, we need tx < 31 and ty < 15.  Threads on the right column
    // (tx == 31) and bottom row (ty == 15) only contribute their corner
    // during Phase 1 — they don't process a pixel in Phase 2.
    const bool is_pixel_thread = (tx < KABSCH_BLOCK_X - 1) && (ty < KABSCH_BLOCK_Y - 1);
    const bool pixel_in_image = is_pixel_thread && (gx < static_cast<int>(width))
                                && (gy < static_cast<int>(height));

    // ── Reflection Processing Loop ──────────────────────────────────
    // Process each reflection: compute corner flags, sync, then accumulate pixels
    for (size_t r = 0; r < s_num_block_refl; ++r) {
        const size_t refl_idx = s_block_refl[r];
        const Vector3D &s1_c = d_s1_vectors[refl_idx];
        const scalar_t phi_c = d_phi_values[refl_idx];
        const BoundingBoxExtents &bbox = d_bboxes[refl_idx];

        // Every thread writes its corner's foreground flag to shared memory.
        // Per-corner bbox check avoids needless Kabsch computation.
        const bool in_bbox = corner_valid && (gx >= bbox.x_min && gx <= bbox.x_max)
                             && (gy >= bbox.y_min && gy <= bbox.y_max);

        if (in_bbox) {
            scalar_t s1_len;  // unused here, required by pixel_to_kabsch

            // Lower z-face (phi_low)
            Vector3D eps_lo =
              pixel_to_kabsch(s0, s1_c, phi_c, s_pixel, phi_low, rot_axis, s1_len);
            s_fg[0][ty][tx] = is_foreground(eps_lo, delta_b, delta_m);

            // Upper z-face (phi_high)
            Vector3D eps_hi =
              pixel_to_kabsch(s0, s1_c, phi_c, s_pixel, phi_high, rot_axis, s1_len);
            s_fg[1][ty][tx] = is_foreground(eps_hi, delta_b, delta_m);
        } else {
            s_fg[0][ty][tx] = false;
            s_fg[1][ty][tx] = false;
        }

        __syncthreads();

        // Interior threads check all 8 voxel corners for foreground classification
        if (pixel_in_image) {
            // Union of all 8 corners: any foreground -> pixel is foreground
            const bool any_fg = s_fg[0][ty][tx] || s_fg[0][ty][tx + 1]
                                || s_fg[0][ty + 1][tx] || s_fg[0][ty + 1][tx + 1]
                                || s_fg[1][ty][tx] || s_fg[1][ty][tx + 1]
                                || s_fg[1][ty + 1][tx] || s_fg[1][ty + 1][tx + 1];

            // Only accumulate if the pixel centre is within the bbox
            // (pixels outside the bbox are irrelevant to this reflection)
            const bool px_in_bbox = (gx >= bbox.x_min && gx < bbox.x_max)
                                    && (gy >= bbox.y_min && gy < bbox.y_max);

            if (px_in_bbox) {
                const accumulator_t pixel_value =
                  static_cast<accumulator_t>(image(gx, gy));

                if (any_fg) {
                    atomicAdd(&d_foreground_sum[refl_idx], pixel_value);
                    atomicAdd(&d_foreground_count[refl_idx], 1u);
                } else {
                    atomicAdd(&d_background_sum[refl_idx], pixel_value);
                    atomicAdd(&d_background_count[refl_idx], 1u);
                }
            }
        }

        __syncthreads();  // Barrier before next reflection overwrites s_fg
    }
}

/**
 * @brief Host wrapper function for image-based Kabsch computation
 *
 * Launches the 8-corner Kabsch kernel using overlapping tiles.  Each block
 * of KABSCH_BLOCK_X × KABSCH_BLOCK_Y threads covers that many corners,
 * processing (KABSCH_BLOCK_X-1) × (KABSCH_BLOCK_Y-1) complete pixels.
 */
void compute_kabsch_transform(pixel_t *d_image,
                              size_t image_pitch,
                              uint32_t width,
                              uint32_t height,
                              int image_num,
                              const scalar_t *d_matrix,
                              scalar_t wavelength,
                              scalar_t osc_start,
                              scalar_t osc_width,
                              int image_range_start,
                              Vector3D s0,
                              Vector3D rot_axis,
                              const Vector3D *d_s1_vectors,
                              const scalar_t *d_phi_values,
                              const BoundingBoxExtents *d_bboxes,
                              const size_t *d_reflection_indices,
                              size_t num_reflections_this_image,
                              scalar_t delta_b,
                              scalar_t delta_m,
                              accumulator_t *d_foreground_sum,
                              uint32_t *d_foreground_count,
                              accumulator_t *d_background_sum,
                              uint32_t *d_background_count,
                              cudaStream_t stream) {
    // Configure kernel launch parameters.
    // Every block is always the full 32×16 = 512 threads.
    dim3 blockDim(KABSCH_BLOCK_X, KABSCH_BLOCK_Y);

    // Each block processes (BLOCK-1) complete pixels per dimension:
    //   31 pixels in x, 15 pixels in y.
    // So the number of blocks needed is ceil(image_pixels / pixels_per_block):
    //   gridDim.x = ceil(width  / 31)
    //   gridDim.y = ceil(height / 15)
    // Edge blocks may extend past the image boundary — those threads
    // simply fail the bounds check in the kernel and write false / skip.
    dim3 gridDim(ceil_div(width, static_cast<uint32_t>(KABSCH_BLOCK_X - 1)),
                 ceil_div(height, static_cast<uint32_t>(KABSCH_BLOCK_Y - 1)));

    // Launch the summation integration kernel
    kabsch_transform<<<gridDim, blockDim, 0, stream>>>(d_image,
                                                       image_pitch,
                                                       width,
                                                       height,
                                                       image_num,
                                                       d_matrix,
                                                       wavelength,
                                                       osc_start,
                                                       osc_width,
                                                       image_range_start,
                                                       s0,
                                                       rot_axis,
                                                       d_s1_vectors,
                                                       d_phi_values,
                                                       d_bboxes,
                                                       d_reflection_indices,
                                                       num_reflections_this_image,
                                                       delta_b,
                                                       delta_m,
                                                       d_foreground_sum,
                                                       d_foreground_count,
                                                       d_background_sum,
                                                       d_background_count);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
          fmt::format("Kernel launch failed: {}", cudaGetErrorString(err)));
    }
}

// DEPRECATED
/**
 * @brief CUDA kernel to compute pixel-to-Kabsch transformations for
 * multiple voxels
 * 
 * This kernel transforms voxel coordinates from reciprocal space to
 * Kabsch coordinates.
 * Each thread processes one voxel, computing its Kabsch coordinates
 * relative to a reflection center.
 * 
 * @param s_pixels Array of s_pixel vectors (different for each voxel)
 * @param phi_pixels Array of phi_pixel angles (different for each voxel)
 * @param s1_c Reflection center s1 vector (same for all voxels in this batch)
 * @param phi_c Reflection center phi angle (same for all voxels in this batch)
 * @param s0 Initial scattering vector
 * @param rot_axis Rotation axis vector
 * @param eps_array Output array for Kabsch coordinates
 * @param s1_len_array Output array for s1 lengths
 * @param n Number of voxels to process
 */
__global__ void kabsch_transform_flat(const Vector3D *__restrict__ s_pixels,
                                      const scalar_t *__restrict__ phi_pixels,
                                      Vector3D s1_c,
                                      scalar_t phi_c,
                                      Vector3D s0,
                                      Vector3D rot_axis,
                                      Vector3D *eps_array,
                                      scalar_t *s1_len_array,
                                      size_t n) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // Bounds check

    // Get input data for this voxel
    Vector3D s_pixel = s_pixels[i];
    scalar_t phi_pixel = phi_pixels[i];

    // Compute Kabsch transformation using the device function
    scalar_t s1_len;
    Vector3D eps =
      pixel_to_kabsch(s0, s1_c, phi_c, s_pixel, phi_pixel, rot_axis, s1_len);

    // Store results
    eps_array[i] = eps;
    s1_len_array[i] = s1_len;
}

/**
 * @brief Host wrapper function for voxel Kabsch computation
 * 
 * @param h_s_pixels Host array of s_pixel vectors (different for each voxel)
 * @param h_phi_pixels Host array of phi_pixel angles (different for each voxel)
 * @param s1_c Reflection center s1 vector (same for all voxels in this reflection)
 * @param phi_c Reflection center phi angle (same for all voxels in this reflection)
 * @param s0 Initial scattering vector
 * @param rot_axis Rotation axis vector
 * @param h_eps Host array to store output Kabsch coordinates
 * @param h_s1_len Host array to store output s1 lengths
 * @param n Number of voxels
 */
void compute_kabsch_transform(const Vector3D *h_s_pixels,
                              const scalar_t *h_phi_pixels,
                              Vector3D s1_c,
                              scalar_t phi_c,
                              Vector3D s0,
                              Vector3D rot_axis,
                              Vector3D *h_eps,
                              scalar_t *h_s1_len,
                              size_t n) {
    // Create RAII device buffers
    DeviceBuffer<Vector3D> d_s_pixels(n);    // Input s_pixel vectors
    DeviceBuffer<scalar_t> d_phi_pixels(n);  // Input phi_pixel angles
    DeviceBuffer<Vector3D> d_eps(n);         // Output Kabsch coordinates
    DeviceBuffer<scalar_t> d_s1_len(n);      // Output s1 lengths

    // Copy input data from host to device
    d_s_pixels.assign(h_s_pixels);
    d_phi_pixels.assign(h_phi_pixels);

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    kabsch_transform_flat<<<blocksPerGrid, threadsPerBlock>>>(d_s_pixels.data(),
                                                              d_phi_pixels.data(),
                                                              s1_c,
                                                              phi_c,
                                                              s0,
                                                              rot_axis,
                                                              d_eps.data(),
                                                              d_s1_len.data(),
                                                              n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
          fmt::format("Kernel launch failed: {}", cudaGetErrorString(err)));
    }

    // Wait for kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
          fmt::format("Kernel execution failed: {}", cudaGetErrorString(err)));
    }

    // Copy results back to host
    d_eps.extract(h_eps);
    d_s1_len.extract(h_s1_len);
}