/**
 * @file kabsch.cuh
 * @brief Header file for CUDA Kabsch coordinate transformation functions
 * 
 * This header declares the interface for GPU-accelerated coordinate transformations
 * used in the Kabsch algorithm for data processing. The functions
 * convert pixel coordinates from reciprocal space into a geometry-invariant local
 * coordinate system for efficient reflection profile integration.
 * 
 * The Kabsch transformation creates a local coordinate system defined by:
 * - e₁: Perpendicular to the scattering plane (s₁ᶜ × s₀)
 * - e₂: Within the scattering plane, orthogonal to e₁ (s₁ᶜ × e₁)
 * - e₃: Bisects the incident and diffracted beam directions (s₁ᶜ + s₀)
 * 
 * @author Dimitrios Vlachos
 * @date 2025
 * @see kabsch.cu for implementation details
 * @see Vector3D for mathematical vector operations
 * @see DeviceBuffer for GPU memory management
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "h5read.h"
#include "integrator.cuh"
#include "integrator/background.hpp"
#include "integrator/background_pool.hpp"
#include "integrator/extent.hpp"
#include "math/device_precision.cuh"
#include "math/math_utils.cuh"
#include "math/vector3d.cuh"

/// CUDA block width (threads per block) for the shoebox-parallel Kabsch kernel.
/// Each block strides this many threads over its shoebox's pixels, so it makes
/// ceil(w*h / KABSCH_THREADS) passes. A block-size sweep found the kernel is
/// compute-bound and its time is flat within ~8% from 32 to 512, so this is a
/// fixed constant, not a tuning knob. Must be a multiple of the 32-lane warp;
/// note 1024 exceeds the per-block register budget at this kernel's register
/// pressure and fails to launch.
constexpr uint32_t KABSCH_THREADS = 128;

using namespace fastvec;
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
           - (t0 / cos_t + scalar_t(1.0) / mu) * CUDA_EXP(-mu * t0 / cos_t);
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
 * @param cx Corner x-coordinate in pixels.
 * @param cy Corner y-coordinate in pixels.
 * @param d_matrix Detector matrix (3x3 flattened) for mm to lab conversion.
 * @param wavelength Beam wavelength in Å.
 * @param det_params Detector geometry used for the pixel to mm conversion.
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
 * @brief Host wrapper function for image-based Kabsch computation
 * 
 * Launches the shoebox-parallel Kabsch kernel, one block per reflection on
 * this image, to classify each shoebox pixel as foreground or background and
 * atomically accumulate intensities for summation integration. A masked or
 * out-of-image foreground pixel fails the reflection via d_success.
 *
 * @param d_image Device pointer to image data (pitched allocation)
 * @param image_pitch Pitch in bytes of the device image allocation
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param image_num Current image/frame number
 * @param d_matrix Detector matrix (3x3 flattened) for pixel to lab conversion
 * @param wavelength Beam wavelength for s_pixel normalization
 * @param det_params Detector geometry used for the pixel to mm conversion
 * @param osc_start Oscillation start angle (degrees)
 * @param osc_width Oscillation width per image (degrees)
 * @param image_range_start First image number in the scan
 * @param s0 Incident beam vector
 * @param rot_axis Rotation axis vector
 * @param d_s1_vectors Device array of s1_c vectors for each reflection
 * @param d_phi_values Device array of phi_c values for each reflection
 * @param d_bboxes Device array of BoundingBoxExtents structs for each reflection
 * @param d_reflection_indices Device array of reflection indices for this image
 * @param num_reflections_this_image Number of reflections touching this image
 * @param d_mask Detector mask, flat width*height indexed gy*width + gx (non-zero = valid, 0 = masked)
 * @param max_corner_tile_corners Corner slots for the shared-memory corner tile:
 *        the maximum (w+1)*(h+1) over this image's reflection shoeboxes. Sizes
 *        the kernel's dynamic shared memory (one byte per corner), clamped
 *        internally to the device per-block limit; a shoebox beyond the clamp
 *        falls back to per-pixel corner recompute
 * @param delta_b Foreground extent in e₁/e₂ directions (n_sigma × σ_D), radians
 * @param delta_m Foreground extent in e₃ direction (n_sigma × σ_M), radians
 * @param algorithm Foreground model used to classify each pixel
 * @param d_foreground_sum Device array to accumulate foreground intensities per reflection
 * @param d_foreground_count Device array to count foreground pixels per reflection
 * @param background_pool Shared device background pool, keyed on (reflection, value)
 * @param d_background_spill Device per-reflection count of pixels dropped by a full background pool
 * @param d_intensity_times_x Device array accumulating intensity·(2gx+1) per reflection (centre-of-mass)
 * @param d_intensity_times_y Device array accumulating intensity·(2gy+1) per reflection (centre-of-mass)
 * @param d_intensity_times_z Device array accumulating intensity·(2z+1) per reflection (centre-of-mass)
 * @param d_success Per-reflection success flag, cleared on a masked or out-of-image foreground pixel
 * @param stream CUDA stream for async execution
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
                              fastvec::Vector3D s0,
                              fastvec::Vector3D rot_axis,
                              const fastvec::Vector3D *d_s1_vectors,
                              const scalar_t *d_phi_values,
                              const BoundingBoxExtents *d_bboxes,
                              const size_t *d_reflection_indices,
                              size_t num_reflections_this_image,
                              const uint8_t *d_mask,
                              uint32_t max_corner_tile_corners,
                              scalar_t delta_b,
                              scalar_t delta_m,
                              FGAlgorithm algorithm,
                              accumulator_t *d_foreground_sum,
                              uint32_t *d_foreground_count,
                              BackgroundPool background_pool,
                              uint32_t *d_background_spill,
                              unsigned long long *d_intensity_times_x,
                              unsigned long long *d_intensity_times_y,
                              unsigned long long *d_intensity_times_z,
                              uint8_t *d_success,
                              cudaStream_t stream);
