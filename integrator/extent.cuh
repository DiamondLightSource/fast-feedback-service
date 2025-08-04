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
 * @brief Host wrapper function for computing Kabsch bounding boxes on GPU
 * 
 * This function provides a high-level interface for computing 3D spatial extents
 * of X-ray diffraction reflections using GPU acceleration. It manages all CUDA
 * memory operations, kernel launch parameters, and error handling.
 * 
 * The function performs the following operations:
 * 1. Allocates GPU memory buffers for input data and results
 * 2. Transfers input data from host to device memory
 * 3. Launches the CUDA kernel with optimal thread/block configuration
 * 4. Handles CUDA errors and provides detailed error messages
 * 5. Transfers results back to host memory in the expected format
 * 
 * Memory Management:
 * - Uses RAII DeviceBuffer for automatic memory cleanup
 * - Handles large datasets efficiently with bulk transfers
 * - Converts between internal BoundingBox format and flat array output
 * 
 * Error Handling:
 * - Validates kernel launch success
 * - Ensures kernel execution completes without errors
 * - Provides descriptive error messages using fmt formatting
 * 
 * @param h_s1_vectors Host array of s1 diffraction vectors for all reflections
 * @param h_phi_positions Host array of phi rotation angles for all reflections
 * @param s0 Incident beam vector (normalized)
 * @param rot_axis Crystal rotation axis vector
 * @param sigma_b Beam divergence standard deviation (radians)
 * @param sigma_m Crystal mosaicity standard deviation (radians)
 * @param n_sigma Number of standard deviations for extent calculation
 * @param sigma_b_multiplier Additional scaling factor for beam divergence
 * @param osc_start Oscillation start angle (degrees)
 * @param osc_width Oscillation width per frame (degrees)
 * @param image_range_start Starting frame/image number
 * @param image_range_end Ending frame/image number
 * @param wavelength X-ray wavelength (Angstroms)
 * @param d_matrix_inv_flat Inverse detector D-matrix as 9-element row-major array
 * @param h_bounding_boxes Host array to store output bounding boxes (6 values per reflection: x_min, x_max, y_min, y_max, z_min, z_max)
 * @param num_reflections Total number of reflections to process
 * 
 * @throws std::runtime_error If CUDA kernel launch or execution fails
 * 
 * @note The output array h_bounding_boxes must be pre-allocated with size num_reflections * 6
 * @note All input arrays must contain exactly num_reflections elements
 * @note The d_matrix_inv_flat array must contain exactly 9 elements (3x3 matrix in row-major order)
 */
void compute_kabsch_bounding_boxes_cuda(const fastvec::Vector3D* const h_s1_vectors,
                                        const scalar_t* const h_phi_positions,
                                        const fastvec::Vector3D s0,
                                        const fastvec::Vector3D rot_axis,
                                        const scalar_t sigma_b,
                                        const scalar_t sigma_m,
                                        const scalar_t n_sigma,
                                        const scalar_t sigma_b_multiplier,
                                        const scalar_t osc_start,
                                        const scalar_t osc_width,
                                        const int image_range_start,
                                        const int image_range_end,
                                        const scalar_t wavelength,
                                        const scalar_t* const d_matrix_inv_flat,
                                        scalar_t* const h_bounding_boxes,
                                        const size_t num_reflections);