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
#include "integrator/extent.hpp"
#include "math/device_precision.cuh"
#include "math/vector3d.cuh"

/// CUDA block width (threads per block) for the shoebox-parallel Kabsch kernel.
/// Each block strides this many threads over its shoebox's pixels, so it makes
/// ceil(w*h / KABSCH_THREADS) passes. A block-size sweep found the kernel is
/// compute-bound and its time is flat within ~8% from 32 to 512, so this is a
/// fixed constant, not a tuning knob. Must be a multiple of the 32-lane warp;
/// note 1024 exceeds the per-block register budget at this kernel's register
/// pressure and fails to launch.
constexpr uint32_t KABSCH_THREADS = 128;

/**
 * @brief Host wrapper function for image-based Kabsch computation
 * 
 * Launches a kernel over the padded grid (image plus any bbox overflow past
 * the detector edge) to compute Kabsch coordinates for pixels that fall within
 * reflection bounding boxes, classifying each pixel as foreground or background
 * and atomically accumulating intensities for summation integration. A masked
 * or out-of-image foreground pixel fails the reflection via d_success.
 *
 * @param d_image Device pointer to image data (pitched allocation)
 * @param image_pitch Pitch in bytes of the device image allocation
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param image_num Current image/frame number
 * @param d_matrix Detector matrix (3x3 flattened) for pixel to lab conversion
 * @param wavelength Beam wavelength for s_pixel normalization
 * @param osc_start Oscillation start angle (radians)
 * @param osc_width Oscillation width per image (radians)
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
 * @param d_foreground_sum Device array to accumulate foreground intensities per reflection
 * @param d_foreground_count Device array to count foreground pixels per reflection
 * @param d_background_hist Device per-reflection background histogram, num_reflections*NUM_BG_BINS bins (one per integer pixel value)
 * @param d_background_overflow Device per-reflection count of background pixels with value >= NUM_BG_BINS (high tail)
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
                              uint32_t *d_background_hist,
                              uint32_t *d_background_overflow,
                              unsigned long long *d_intensity_times_x,
                              unsigned long long *d_intensity_times_y,
                              unsigned long long *d_intensity_times_z,
                              uint8_t *d_success,
                              cudaStream_t stream);
