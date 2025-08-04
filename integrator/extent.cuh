/**
 * @file extent.cuh
 * @brief Header file for CUDA Kabsch bounding box extent computation functions
 * 
 * This header declares the interface for GPU-accelerated bounding box computations
 * used in X-ray diffraction reflection processing. The functions compute 3D spatial
 * extents for reflections using the Kabsch coordinate transformation to account for
 * beam divergence, crystal mosaicity, and detector geometry.
 * 
 * The bounding box algorithm creates spatial extents by:
 * - Projecting beam divergence uncertainties onto Kabsch basis vectors
 * - Computing corner positions on the Ewald sphere
 * - Converting to detector pixel coordinates
 * - Calculating rotation range from mosaicity parameters
 * 
 * @author Dimitrios Vlachos
 * @date 2025
 * @see extent.cu for implementation details
 * @see kabsch.cuh for coordinate transformation functions
 * @see Vector3D for mathematical vector operations
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "math/device_precision.cuh"
#include "math/vector3d.cuh"

/**
 * @brief Host wrapper function for computing Kabsch bounding boxes with detector parameters
 * 
 * This function provides correct detector coordinate transformations including
 * parallax correction by accepting all necessary detector parameters.
 * 
 * @param h_s1_vectors Host array of s1 diffraction vectors for all reflections
 * @param h_phi_positions Host array of phi rotation angles for all reflections
 * @param s0 Incident beam vector (normalized)
 * @param rot_axis Crystal rotation axis vector
 * @param sigma_b Beam divergence standard deviation (radians)
 * @param sigma_m Crystal mosaicity standard deviation (radians)
 * @param osc_start Oscillation start angle (degrees)
 * @param osc_width Oscillation width per frame (degrees)
 * @param image_range_start Starting frame/image number
 * @param image_range_end Ending frame/image number
 * @param wavelength X-ray wavelength (Angstroms)
 * @param d_matrix_inv_flat Inverse detector D-matrix as 9-element row-major array
 * @param pixel_size Array of pixel sizes [x_size, y_size] in mm
 * @param parallax_correction Whether to apply parallax correction
 * @param mu Absorption coefficient (if parallax correction enabled)
 * @param thickness Detector thickness (if parallax correction enabled)
 * @param fast_axis Detector fast axis vector
 * @param slow_axis Detector slow axis vector
 * @param origin Detector origin vector
 * @param h_bounding_boxes Host array to store output bounding boxes
 * @param num_reflections Total number of reflections to process
 * @param n_sigma Number of standard deviations for extent calculation
 * @param sigma_b_multiplier Additional scaling factor for beam divergence
 */
void compute_bbox_extent(const fastvec::Vector3D* const h_s1_vectors,
                         const scalar_t* const h_phi_positions,
                         const fastvec::Vector3D s0,
                         const fastvec::Vector3D rot_axis,
                         const scalar_t sigma_b,
                         const scalar_t sigma_m,
                         const scalar_t osc_start,
                         const scalar_t osc_width,
                         const int image_range_start,
                         const int image_range_end,
                         const scalar_t wavelength,
                         const scalar_t* const d_matrix_inv_flat,
                         const scalar_t* const pixel_size,
                         const bool parallax_correction,
                         const scalar_t mu,
                         const scalar_t thickness,
                         const fastvec::Vector3D fast_axis,
                         const fastvec::Vector3D slow_axis,
                         const fastvec::Vector3D origin,
                         scalar_t* const h_bounding_boxes,
                         const size_t num_reflections,
                         const scalar_t n_sigma = 3.0,
                         const scalar_t sigma_b_multiplier = 2.0);
