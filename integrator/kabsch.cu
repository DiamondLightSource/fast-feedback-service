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
 * 
 * TODO:
 * 2. Fix unit test
 *     - Remove deprecated kernel
 *     - Refactor unit test to utilise the new kernel and host wrapper
 * 
 * 3. Dials-data?
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

#include "cuda_common.hpp"
#include "device_common.cuh"
#include "h5read.h"
#include "integrator.cuh"
#include "integrator/extent.hpp"
#include "kabsch.cuh"
#include "math/math_utils.cuh"

using namespace fastvec;

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
    // Precompute inverse squares to avoid repeated division.
    // δ_B² appears in two terms (ε₁² and ε₂²); compute 1/δ_B² once and
    // multiply both. Division is much more expensive than
    // multiplication on the GPU. Rather than dividing three times, we
    // compute the inverse squares once and multiply, which should be
    // more efficient.
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
    // Vector3D s1_phi_prime = s1_c + e3 * dphi; // Temporarily disable improved calculation

    // Difference vector between pixel's s′ and the φ′-adjusted centroid
    // Vector3D deltaS = s_pixel - s1_phi_prime; // Temporarily disable improved calculation
    Vector3D deltaS = s_pixel - s1_c;  // Temporarily disable improved calculation

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
 * @brief Mean absorption path length through a flat sensor
 *
 *   o = (1/μ) − (t₀/cos_θ + 1/μ) · exp(−μ · t₀ / cos_θ)
 *
 * @param mu Linear attenuation coefficient μ (mm⁻¹)
 * @param t0 Sensor thickness t₀ (mm)
 * @param s1 Diffracted unit vector at the pixel
 * @param fast Panel fast-axis direction f̂
 * @param slow Panel slow-axis direction ŝ
 * @param origin Panel origin (mm)
 * @return Absorption depth o (mm)
 */
__device__ __forceinline__ scalar_t attenuation_length(scalar_t mu,
                                                       scalar_t t0,
                                                       Vector3D s1,
                                                       Vector3D fast,
                                                       Vector3D slow,
                                                       Vector3D origin) {
    // Panel normal n̂ = f̂ × ŝ, oriented toward the X-ray source
    Vector3D normal = cross(fast, slow);
    scalar_t distance = dot(origin, normal);
    if (distance < scalar_t(0.0)) {
        normal = -normal;
    }

    scalar_t cos_t = dot(s1, normal);

    return (scalar_t(1.0) / mu)
           - (t0 / cos_t + scalar_t(1.0) / mu) * exp(-mu * t0 / cos_t);
}

/**
 * @brief Convert a pixel position to millimetre coordinates, optionally
 *        applying the sensor parallax correction
 *
 *   x_mm = gx · px_size[0],  y_mm = gy · px_size[1]
 *   c₁ = x_mm − (s₁ · f̂) · o
 *   c₂ = y_mm − (s₁ · ŝ) · o
 *
 * GPU port of dx2::Panel::px_to_mm (detector.cxx).
 *
 * @param gx Pixel x index (fast axis)
 * @param gy Pixel y index (slow axis)
 * @param det Detector geometry and sensor parameters
 * @param out_x Output x coordinate in mm
 * @param out_y Output y coordinate in mm
 */
__device__ __forceinline__ void px_to_mm(int gx,
                                         int gy,
                                         const DetectorParameters &det,
                                         scalar_t &out_x,
                                         scalar_t &out_y) {
    scalar_t x1 = static_cast<scalar_t>(gx) * det.pixel_size[0];
    scalar_t x2 = static_cast<scalar_t>(gy) * det.pixel_size[1];

    if (!det.parallax_correction) {
        out_x = x1;
        out_y = x2;
        return;
    }

    Vector3D s1 = det.origin + det.fast_axis * x1 + det.slow_axis * x2;
    s1 = normalized(s1);

    scalar_t o = attenuation_length(
      det.mu, det.thickness, s1, det.fast_axis, det.slow_axis, det.origin);

    out_x = x1 - dot(s1, det.fast_axis) * o;
    out_y = x2 - dot(s1, det.slow_axis) * o;
}

/**
 * @brief Scattered beam wavevector s1 (magnitude 1/λ) at a detector corner
 *        (cx, cy).
 *
 * The px -> reciprocal-space mapping is pure geometry, so it holds fine past
 * the detector edge. Corners off the image still give real Kabsch coordinates
 * and classify the same as any other, so we don't guard for them here.
 *
 * @param wavelength Beam wavelength in Å.
 * @return s1 in lab frame, units of 1/Å (lies on the Ewald sphere).
 */
__device__ __forceinline__ Vector3D
corner_s_pixel(int cx,
               int cy,
               const scalar_t *d_matrix,
               scalar_t wavelength,
               const DetectorParameters &det_params) {
    scalar_t cx_mm, cy_mm;
    px_to_mm(cx, cy, det_params, cx_mm, cy_mm);

    Vector3D lab_coord =
      make_vector3d(d_matrix[0] * cx_mm + d_matrix[1] * cy_mm + d_matrix[2],
                    d_matrix[3] * cx_mm + d_matrix[4] * cy_mm + d_matrix[5],
                    d_matrix[6] * cx_mm + d_matrix[7] * cy_mm + d_matrix[8]);
    return normalized(lab_coord) / wavelength;
}

/**
 * @brief Classify a single pixel as foreground for one reflection.
 *
 * A pixel is foreground if ANY of its four voxel corners {px,px+1}×{py,py+1}
 * is inside the reflection profile in ANY evaluated z-slice. Each thread
 * computes its own pixel's corners directly, with no shared-memory corner cache
 * and no inter-thread barrier. The corner Kabsch values are deterministic in
 * (corner, φ), so neighbouring threads that share a corner compute the same
 * value independently. This recomputes each interior corner up to four times, a
 * deliberate trade: the kernel is latency-bound rather than compute-bound (see
 * kabsch_transform), so the redundant ALU is effectively free, whereas caching
 * the corners would need a barrier that is not. It returns on the first corner
 * that lands inside the profile, so foreground pixels bail early and only
 * fully-background ones pay for all four corners.
 *
 * Ellipsoid evaluates three z-slices (phi_low, phi_high, and phi_c when the
 * centre falls in this slice); DIALS evaluates a single 2D ellipse at phi_low,
 * ignoring ε₃.
 *
 * @tparam Algo Foreground model: Ellipsoid (3D) or DIALS (2D ellipse).
 * @param px Pixel fast-axis index (lower corner), in pixels.
 * @param py Pixel slow-axis index (lower corner), in pixels.
 * @param s0 Incident beam vector (s₀), units of 1/Å.
 * @param s1_c Predicted diffracted vector at the reflection centre (s₁ᶜ), units of 1/Å.
 * @param phi_c Rotation angle at the reflection centre (φᶜ), in radians.
 * @param phi_low Rotation angle at the lower z-face (φ), in radians.
 * @param phi_high Rotation angle at the upper z-face (φ), in radians.
 * @param centre_in_z_slice Whether φᶜ falls in this image's z-slice (Ellipsoid centre slice).
 * @param rot_axis Unit goniometer rotation axis vector (m₂).
 * @param d_matrix Detector coordinate transformation matrix (3x3).
 * @param wavelength X-ray wavelength in Å.
 * @param det_params Detector geometry parameters (pixel size, parallax, etc.).
 * @param delta_b Foreground extent in e₁/e₂ directions (n_sigma × σ_D).
 * @param delta_m Foreground extent in e₃ direction (n_sigma × σ_M).
 * @return true if the pixel is foreground for this reflection, false otherwise.
 */
template <FGAlgorithm Algo>
__device__ __forceinline__ bool pixel_is_foreground(
  int px,
  int py,
  const Vector3D &s0,
  const Vector3D &s1_c,
  scalar_t phi_c,
  scalar_t phi_low,
  scalar_t phi_high,
  bool centre_in_z_slice,
  const Vector3D &rot_axis,
  const scalar_t *d_matrix,
  scalar_t wavelength,
  const DetectorParameters &det_params,
  scalar_t delta_b,
  scalar_t delta_m) {
#pragma unroll
    for (int dy = 0; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = 0; dx <= 1; ++dx) {
            const Vector3D s_pixel =
              corner_s_pixel(px + dx, py + dy, d_matrix, wavelength, det_params);
            scalar_t s1_len;  // unused here, required by pixel_to_kabsch

            // `if constexpr` resolves at compile time; only the chosen branch is
            // emitted into each kernel specialization, so the unused algorithm's
            // code (and its register footprint) is gone.
            if constexpr (Algo == FGAlgorithm::Ellipsoid) {
                // Lower z-face (phi_low)
                Vector3D eps_lo =
                  pixel_to_kabsch(s0, s1_c, phi_c, s_pixel, phi_low, rot_axis, s1_len);
                if (is_foreground(eps_lo, delta_b, delta_m)) return true;

                // Upper z-face (phi_high)
                Vector3D eps_hi =
                  pixel_to_kabsch(s0, s1_c, phi_c, s_pixel, phi_high, rot_axis, s1_len);
                if (is_foreground(eps_hi, delta_b, delta_m)) return true;

                // Centre z-slice: evaluate at phi_c where ε₃ = 0
                if (centre_in_z_slice) {
                    Vector3D eps_c = pixel_to_kabsch(
                      s0, s1_c, phi_c, s_pixel, phi_c, rot_axis, s1_len);
                    if (is_foreground(eps_c, delta_b, delta_m)) return true;
                }
            } else {
                // DIALS: single 2D ellipse, evaluated at phi_low; ε₃ ignored
                Vector3D eps =
                  pixel_to_kabsch(s0, s1_c, phi_c, s_pixel, phi_low, rot_axis, s1_len);
                scalar_t inv_delta_b_sq = scalar_t(1.0) / (delta_b * delta_b);
                scalar_t ellipsoid_val =
                  (eps.x * eps.x + eps.y * eps.y) * inv_delta_b_sq;
                if (ellipsoid_val <= scalar_t(1.0)) return true;
            }
        }
    }
    return false;
}

/**
 * @brief Shoebox-parallel Kabsch foreground/background classification kernel.
 *
 * One CUDA block per reflection shoebox on the current image
 * (gridDim.x == num_reflections_this_image; blockIdx.x indexes
 * d_reflection_indices). Threads in the block stride over the shoebox's w×h
 * pixels; each thread classifies its own pixel via its four voxel corners
 * (see pixel_is_foreground) and atomically accumulates into the per-reflection
 * foreground/background buffers.
 *
 * There is no shared memory and no __syncthreads(): every lane works on a real
 * shoebox pixel, and threads past the last shoebox pixel simply retire. With no
 * barrier and no idle background lanes, warps never stall waiting on divergent
 * siblings, which is the point of classifying per shoebox rather than over the
 * full detector frame.
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
 * @param num_reflections_this_image Number of reflections (== gridDim.x)
 * @param d_mask Detector mask, flat width*height
 * @param delta_b Foreground extent in e₁/e₂ directions (n_sigma × σ_D)
 * @param delta_m Foreground extent in e₃ direction (n_sigma × σ_M)
 * @param d_foreground_sum Output: accumulated foreground intensities per reflection
 * @param d_foreground_count Output: foreground pixel counts per reflection
 * @param d_background_hist Output: per-reflection background histogram (num_reflections*NUM_BG_BINS bins, one per integer value)
 * @param d_background_overflow Output: per-reflection count of background pixels with value >= NUM_BG_BINS
 * @param d_intensity_times_x Output: intensity·(2gx+1) per reflection (centre-of-mass)
 * @param d_intensity_times_y Output: intensity·(2gy+1) per reflection (centre-of-mass)
 * @param d_intensity_times_z Output: intensity·(2z+1) per reflection (centre-of-mass)
 * @param d_success Output: per-reflection success flag, cleared on a masked or out-of-image foreground pixel
 */
template <FGAlgorithm Algo>
__global__ void kabsch_transform(pixel_t *d_image,
                                 size_t image_pitch,
                                 uint32_t width,
                                 uint32_t height,
                                 int image_num,
                                 const scalar_t *d_matrix,
                                 scalar_t wavelength,
                                 DetectorParameters det_params,
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
                                 const uint8_t *d_mask,
                                 // Summation integration parameters
                                 scalar_t delta_b,
                                 scalar_t delta_m,
                                 // Output accumulators (atomically updated)
                                 accumulator_t *d_foreground_sum,
                                 uint32_t *d_foreground_count,
                                 uint32_t *d_background_hist,
                                 uint32_t *d_background_overflow,
                                 unsigned long long *d_intensity_times_x,
                                 unsigned long long *d_intensity_times_y,
                                 unsigned long long *d_intensity_times_z,
                                 uint8_t *d_success) {
    // One block per reflection shoebox on this image; blockIdx.x indexes the
    // image-level reflection list directly.
    const size_t block_refl = blockIdx.x;
    if (block_refl >= num_reflections_this_image) return;

    const size_t refl_idx = d_reflection_indices[block_refl];
    const Vector3D s1_c = d_s1_vectors[refl_idx];
    const scalar_t phi_c = d_phi_values[refl_idx];
    const BoundingBoxExtents bbox = d_bboxes[refl_idx];

    // The shoebox's z-range is half-open: [z_min, z_max). The current
    // image_num must be within that range.
    assert(image_num >= bbox.z_min && image_num < bbox.z_max);

    // The voxel's two z-faces, one image apart. slice converts the 0-based
    // image_num to the (z + 1 - image_range_start) frame used everywhere else.
    const int slice = image_num - image_range_start + 1;
    const scalar_t phi_low =
      degrees_to_radians(osc_start + (static_cast<scalar_t>(slice) * osc_width));
    const scalar_t phi_high =
      degrees_to_radians(osc_start + (static_cast<scalar_t>(slice + 1) * osc_width));
    // Whether the reflection centre falls in this image's z-slice (block-uniform).
    const bool centre_in_z_slice = (phi_c >= phi_low && phi_c <= phi_high);

    // Pitched image accessor
    size_t elem_pitch = image_pitch / sizeof(pixel_t);
    PitchedArray2D<pixel_t> image(d_image, &elem_pitch);

    // Shoebox pixel extent. The bbox is half-open in x and y
    // (x_min <= px < x_max), so w×h is exactly the pixel count.
    const int w = bbox.x_max - bbox.x_min;
    const int h = bbox.y_max - bbox.y_min;
    const int npix = w * h;

    // Threads stride over the shoebox pixels. Threads with p >= npix
    // never enter the loop and retire immediately, so there is nothing
    // for idle lanes to stall on.
    for (int p = static_cast<int>(threadIdx.x); p < npix;
         p += static_cast<int>(blockDim.x)) {
        const int px = bbox.x_min + (p % w);
        const int py = bbox.y_min + (p / w);

        // Out-of-image pixels still classify (geometry is valid beyond the
        // edge); a foreground pixel outside the image fails the reflection.
        const bool pixel_in_image = (px >= 0) && (px < static_cast<int>(width))
                                    && (py >= 0) && (py < static_cast<int>(height));

        const bool any_fg = pixel_is_foreground<Algo>(px,
                                                      py,
                                                      s0,
                                                      s1_c,
                                                      phi_c,
                                                      phi_low,
                                                      phi_high,
                                                      centre_in_z_slice,
                                                      rot_axis,
                                                      d_matrix,
                                                      wavelength,
                                                      det_params,
                                                      delta_b,
                                                      delta_m);

        if (any_fg) {
            // Foreground outside the image or on a masked pixel makes the
            // reflection unusable. d_success is only ever cleared, never set,
            // so the concurrent writes need no atomic.
            if (!pixel_in_image || d_mask[static_cast<size_t>(py) * width + px] == 0) {
                d_success[refl_idx] = 0;
            } else {
                const accumulator_t pixel_value =
                  static_cast<accumulator_t>(image(px, py));
                atomicAdd(&d_foreground_sum[refl_idx], pixel_value);
                atomicAdd(&d_foreground_count[refl_idx], 1u);
                // Accumulate intensity·coord for centre-of-mass (xyzobs.px).
                // Baseline uses (x+0.5, y+0.5, z+0.5); we approximate the
                // half-pixel offset by multiplying by 2x+1 etc. and dividing
                // by 2 in finalisation to stay in integer atomics.
                const unsigned long long pv64 =
                  static_cast<unsigned long long>(pixel_value);
                atomicAdd(&d_intensity_times_x[refl_idx],
                          pv64 * static_cast<unsigned long long>(2 * px + 1));
                atomicAdd(&d_intensity_times_y[refl_idx],
                          pv64 * static_cast<unsigned long long>(2 * py + 1));
                atomicAdd(&d_intensity_times_z[refl_idx],
                          pv64 * static_cast<unsigned long long>(2 * image_num + 1));
            }
        } else {
            // Background is counted only for unmasked pixels inside the image;
            // masked or out-of-image background is dropped. Each pixel value
            // increments one bin of this reflection's histogram (one bin per
            // integer count). Values at or above NUM_BG_BINS go to the overflow
            // tail. A negative value is a sentinel/garbage pixel that slipped
            // past the mask (reachable only when pixel_t is 32-bit and the value
            // exceeds INT_MAX); it is not a real background measurement, so it
            // is dropped rather than counted.
            if (pixel_in_image && d_mask[static_cast<size_t>(py) * width + px] != 0) {
                const int bin = static_cast<int>(image(px, py));
                if (bin >= 0 && bin < NUM_BG_BINS) {
                    atomicAdd(&d_background_hist[refl_idx * NUM_BG_BINS + bin], 1u);
                } else if (bin >= NUM_BG_BINS) {
                    atomicAdd(&d_background_overflow[refl_idx], 1u);
                }
            }
        }
    }
}

/**
 * @brief Host wrapper function for image-based Kabsch computation
 *
 * Launches the shoebox-parallel Kabsch kernel: one block of KABSCH_THREADS
 * threads per reflection on this image, striding over that shoebox's pixels.
 */
void compute_kabsch_transform(pixel_t *d_image,
                              size_t image_pitch,
                              uint32_t width,
                              uint32_t height,
                              int image_num,
                              const scalar_t *d_matrix,
                              scalar_t wavelength,
                              DetectorParameters det_params,
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
                              const uint8_t *d_mask,
                              scalar_t delta_b,
                              scalar_t delta_m,
                              FGAlgorithm algorithm,
                              accumulator_t *d_foreground_sum,
                              uint32_t *d_foreground_count,
                              uint32_t *d_background_hist,
                              uint32_t *d_background_overflow,
                              unsigned long long *d_intensity_times_x,
                              unsigned long long *d_intensity_times_y,
                              unsigned long long *d_intensity_times_z,
                              uint8_t *d_success,
                              cudaStream_t stream) {
    // One block per reflection shoebox on this image; nothing to launch
    // if the image has no reflections.
    if (num_reflections_this_image == 0) return;

    // Configure kernel launch parameters.
    dim3 blockDim(KABSCH_THREADS);
    dim3 gridDim(static_cast<uint32_t>(num_reflections_this_image));

    // The kernel is templated on FGAlgorithm rather than taking it as a
    // runtime parameter, for two reasons:
    //   - The classification maths differs (Ellipsoid evaluates three z-slices,
    //     DIALS a single 2D ellipse) and `if constexpr` selects the branch at
    //     compile time.
    //   - A runtime `if` would keep both paths live in every launch, dragging
    //     registers up and occupancy down.
    //
    // The catch is that the algorithm choice comes in at runtime from the
    // CLI, but template parameters have to be compile-time. The lambda
    // fixes this by passing an integral_constant tag that forces a
    // fresh instantiation per call site with `algo` bound to a literal, which
    // then names the right specialisation. The argument list also only has
    // to be written once, which is a nice
    auto launch = [&](auto algo_tag) {
        constexpr FGAlgorithm algo = decltype(algo_tag)::value;
        kabsch_transform<algo>
          <<<gridDim, blockDim, 0, stream>>>(d_image,
                                             image_pitch,
                                             width,
                                             height,
                                             image_num,
                                             d_matrix,
                                             wavelength,
                                             det_params,
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
                                             d_mask,
                                             delta_b,
                                             delta_m,
                                             d_foreground_sum,
                                             d_foreground_count,
                                             d_background_hist,
                                             d_background_overflow,
                                             d_intensity_times_x,
                                             d_intensity_times_y,
                                             d_intensity_times_z,
                                             d_success);
    };
    if (algorithm == FGAlgorithm::Ellipsoid) {
        launch(std::integral_constant<FGAlgorithm, FGAlgorithm::Ellipsoid>{});
    } else {
        launch(std::integral_constant<FGAlgorithm, FGAlgorithm::Dials>{});
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
          fmt::format("Kernel launch failed: {}", cudaGetErrorString(err)));
    }
}
