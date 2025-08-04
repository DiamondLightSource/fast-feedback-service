/**
 * @file extent.cu
 * @brief CUDA implementation for computing Kabsch bounding box extents
 * 
 * This file implements GPU-accelerated computation of spatial extents for X-ray
 * diffraction reflections using the Kabsch coordinate transformation. The bounding
 * boxes define the 3D detector regions where reflection intensity is expected,
 * accounting for beam divergence, crystal mosaicity, and rotation geometry.
 * 
 * Key Features:
 * - GPU-parallel processing of thousands of reflections
 * - Kabsch coordinate system for geometry-invariant calculations
 * - Proper handling of Ewald sphere constraints
 * - Support for both regular and nearly-parallel reflection cases
 * 
 * Mathematical Background:
 * The algorithm projects beam divergence and mosaicity uncertainties onto the
 * detector using the Kabsch basis vectors (e₁, e₂, e₃) to create a local
 * coordinate system that simplifies the geometric calculations.
 * 
 * @author Dimitrios Vlachos
 * @date 2025
 * @see extent.cuh for interface documentation
 * @see kabsch.cu for coordinate transformation implementation
 */

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cuda_common.hpp"
#include "extent.cuh"
#include "math/device_precision.cuh"
#include "math/math_utils.cuh"
#include "math/vector3d.cuh"

using namespace fastvec;

/**
 * @brief Structure to hold bounding box data on GPU
 * 
 * This structure stores the 3D spatial extents of a reflection's predicted
 * intensity distribution on the detector. The x/y coordinates represent pixel
 * positions on the detector face, while z coordinates represent frame/image
 * numbers in the rotation series.
 * 
 * @param x_min Minimum x-coordinate (detector pixel)
 * @param x_max Maximum x-coordinate (detector pixel) 
 * @param y_min Minimum y-coordinate (detector pixel)
 * @param y_max Maximum y-coordinate (detector pixel)
 * @param z_min Minimum z-coordinate (frame/image number)
 * @param z_max Maximum z-coordinate (frame/image number)
 */
struct BoundingBox {
    scalar_t x_min, x_max;
    scalar_t y_min, y_max;
    int z_min, z_max;
};

/**
 * @brief Device function to multiply 3x3 matrix with 3D vector
 * 
 * Performs matrix-vector multiplication: result = M * v, where M is a 3x3 matrix
 * stored in row-major order as a flattened array and v is a 3D vector.
 * The operation computes:
 * result.x = M[0]*v.x + M[1]*v.y + M[2]*v.z
 * result.y = M[3]*v.x + M[4]*v.y + M[5]*v.z  
 * result.z = M[6]*v.x + M[7]*v.y + M[8]*v.z
 * 
 * @param matrix_flat 3x3 matrix stored as 9-element array in row-major order
 * @param vec Input 3D vector to multiply
 * @return Transformed 3D vector result
 */
__device__ Vector3D matrix_vector_multiply(const scalar_t* const matrix_flat,
                                           const Vector3D& vec) {
    return make_vector3d(
      matrix_flat[0] * vec.x + matrix_flat[1] * vec.y + matrix_flat[2] * vec.z,
      matrix_flat[3] * vec.x + matrix_flat[4] * vec.y + matrix_flat[5] * vec.z,
      matrix_flat[6] * vec.x + matrix_flat[7] * vec.y + matrix_flat[8] * vec.z);
}

/**
 * @brief Device function to convert lab coordinates to detector pixel coordinates
 * 
 * Transforms 3D laboratory coordinates to detector pixel coordinates using the
 * inverse detector D-matrix. The D-matrix relates detector pixel coordinates to
 * laboratory millimeter coordinates. This function:
 * 
 * 1. Applies the inverse D-matrix to convert lab coords to detector mm coords
 * 2. Assumes 1:1 mm-to-pixel conversion (simplified - real detectors have pixel sizes)
 * 
 * In a complete implementation, pixel size information would be used to convert
 * from detector mm to actual pixel coordinates.
 * 
 * @param lab_coord 3D coordinates in laboratory reference frame
 * @param d_matrix_inv_flat Inverse D-matrix as 9-element row-major array
 * @return 3D coordinates in detector pixel space
 */
__device__ Vector3D lab_to_detector_pixels(const Vector3D& lab_coord,
                                           const scalar_t* const d_matrix_inv_flat) {
    // Convert lab to detector mm coordinates
    Vector3D detector_mm = matrix_vector_multiply(d_matrix_inv_flat, lab_coord);

    // For simplicity, assume 1:1 mm to pixel conversion
    // In practice, you'd need pixel size information here
    return make_vector3d(detector_mm.x, detector_mm.y, detector_mm.z);
}

/**
 * @brief CUDA kernel to compute Kabsch bounding boxes for multiple reflections
 * 
 * This kernel computes 3D spatial extents for X-ray diffraction reflections using
 * the Kabsch coordinate transformation. Each thread processes one reflection and
 * calculates its bounding box by:
 * 
 * 1. Constructing Kabsch basis vectors (e1, e2) perpendicular to the diffraction vector
 * 2. Projecting beam divergence uncertainties onto these basis vectors
 * 3. Computing corner positions while maintaining Ewald sphere constraints
 * 4. Converting to detector pixel coordinates using the D-matrix
 * 5. Calculating rotation range from mosaicity parameters
 * 
 * Mathematical Background:
 * - Uses Kabsch coordinate system for geometry-invariant calculations
 * - Accounts for beam divergence (sigma_b) and crystal mosaicity (sigma_m)
 * - Handles both regular and nearly-parallel reflection cases
 * - Ensures scattered beams lie on the Ewald sphere
 * 
 * @param s1_vectors Array of s1 diffraction vectors for all reflections
 * @param phi_positions Array of phi rotation angles for all reflections
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
 * @param bounding_boxes Output array to store computed bounding boxes
 * @param num_reflections Total number of reflections to process
 */
__global__ void compute_kabsch_bounding_boxes_kernel(
  const Vector3D* const __restrict__ s1_vectors,
  const scalar_t* const __restrict__ phi_positions,
  const Vector3D s0,
  const Vector3D rot_axis,
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
  BoundingBox* const bounding_boxes,
  const size_t num_reflections) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_reflections) return;

    // Tolerance for nearly-parallel reflections
    constexpr scalar_t ZETA_TOLERANCE = std::numeric_limits<scalar_t>::epsilon() * 1000;

    // Get reflection data
    Vector3D s1_c = s1_vectors[i];
    scalar_t phi_c = phi_positions[i];

    // Construct Kabsch basis vectors
    Vector3D e1 = normalized(cross(s1_c, s0));
    Vector3D e2 = normalized(cross(s1_c, e1));

    scalar_t s1_len = norm(s1_c);

    // Calculate divergence parameters
    scalar_t delta_b = n_sigma * sigma_b * sigma_b_multiplier;
    scalar_t delta_m = n_sigma * sigma_m;

    // Calculate s' vectors at four corners
    scalar_t corner_x[4] = {delta_b, delta_b, -delta_b, -delta_b};
    scalar_t corner_y[4] = {delta_b, -delta_b, delta_b, -delta_b};

    // Initialize min/max extents
    scalar_t min_x = std::numeric_limits<scalar_t>::max();
    scalar_t max_x = std::numeric_limits<scalar_t>::lowest();
    scalar_t min_y = std::numeric_limits<scalar_t>::max();
    scalar_t max_y = std::numeric_limits<scalar_t>::lowest();

    // Process each corner
    for (int corner = 0; corner < 4; corner++) {
        // Project divergences onto Kabsch basis
        Vector3D p =
          (corner_x[corner] * e1 / s1_len) + (corner_y[corner] * e2 / s1_len);

        // Ensure s' lies on Ewald sphere
        scalar_t b = s1_len * s1_len - dot(p, p);
        if (b < 0) continue;  // Skip invalid corners

        scalar_t d = -(dot(p, s1_c) / s1_len) + CUDA_SQRT(b);
        Vector3D s_prime = (d * s1_c / s1_len) + p;

        // Convert to lab coordinates (assuming unit normalization for direction)
        Vector3D lab_coord = s_prime * wavelength;

        // Convert to detector coordinates
        Vector3D detector_coord = lab_to_detector_pixels(lab_coord, d_matrix_inv_flat);

        // Update min/max
        min_x = CUDA_MIN(min_x, detector_coord.x);
        max_x = CUDA_MAX(max_x, detector_coord.x);
        min_y = CUDA_MIN(min_y, detector_coord.y);
        max_y = CUDA_MAX(max_y, detector_coord.y);
    }

    // Calculate z-range using mosaicity
    scalar_t zeta = dot(rot_axis, e1);
    int z_min, z_max;

    if (CUDA_ABS(zeta) > ZETA_TOLERANCE) {
        // Convert angular extents to rotation angles
        scalar_t phi_plus = phi_c + delta_m / zeta;
        scalar_t phi_minus = phi_c - delta_m / zeta;

        // Convert to degrees
        scalar_t phi_plus_deg = radians_to_degrees(phi_plus);
        scalar_t phi_minus_deg = radians_to_degrees(phi_minus);

        // Convert to image numbers
        scalar_t z_plus =
          image_range_start - 1 + ((phi_plus_deg - osc_start) / osc_width);
        scalar_t z_minus =
          image_range_start - 1 + ((phi_minus_deg - osc_start) / osc_width);

        // Clamp to image range
        z_min = std::max(image_range_start,
                         static_cast<int>(CUDA_FLOOR(CUDA_MIN(z_plus, z_minus))));
        z_max = std::min(image_range_end,
                         static_cast<int>(CUDA_CEIL(CUDA_MAX(z_plus, z_minus))));
    } else {
        // Parallel case - span entire range
        z_min = image_range_start;
        z_max = image_range_end;
    }

    // Store result
    bounding_boxes[i].x_min = CUDA_FLOOR(min_x);
    bounding_boxes[i].x_max = CUDA_CEIL(max_x);
    bounding_boxes[i].y_min = CUDA_FLOOR(min_y);
    bounding_boxes[i].y_max = CUDA_CEIL(max_y);
    bounding_boxes[i].z_min = z_min;
    bounding_boxes[i].z_max = z_max;
}

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
void compute_kabsch_bounding_boxes_cuda(const Vector3D* const h_s1_vectors,
                                        const scalar_t* const h_phi_positions,
                                        const Vector3D s0,
                                        const Vector3D rot_axis,
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
                                        const size_t num_reflections) {
    // Create device buffers
    DeviceBuffer<Vector3D> d_s1_vectors(num_reflections);
    DeviceBuffer<scalar_t> d_phi_positions(num_reflections);
    DeviceBuffer<scalar_t> d_matrix_inv(9);  // 3x3 matrix flattened
    DeviceBuffer<BoundingBox> d_bounding_boxes(num_reflections);

    // Copy input data to device
    d_s1_vectors.assign(h_s1_vectors);
    d_phi_positions.assign(h_phi_positions);
    d_matrix_inv.assign(d_matrix_inv_flat);

    // Configure kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_reflections + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    compute_kabsch_bounding_boxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_s1_vectors.data(),
      d_phi_positions.data(),
      s0,
      rot_axis,
      sigma_b,
      sigma_m,
      n_sigma,
      sigma_b_multiplier,
      osc_start,
      osc_width,
      image_range_start,
      image_range_end,
      wavelength,
      d_matrix_inv.data(),
      d_bounding_boxes.data(),
      num_reflections);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(fmt::format("Bounding box kernel launch failed: {}",
                                             cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(fmt::format("Bounding box kernel execution failed: {}",
                                             cudaGetErrorString(err)));
    }

    // Copy results back to host
    std::vector<BoundingBox> temp_boxes(num_reflections);
    d_bounding_boxes.extract(temp_boxes.data());

    // Convert to flat array format (6 values per reflection)
    for (size_t i = 0; i < num_reflections; ++i) {
        h_bounding_boxes[i * 6 + 0] = temp_boxes[i].x_min;
        h_bounding_boxes[i * 6 + 1] = temp_boxes[i].x_max;
        h_bounding_boxes[i * 6 + 2] = temp_boxes[i].y_min;
        h_bounding_boxes[i * 6 + 3] = temp_boxes[i].y_max;
        h_bounding_boxes[i * 6 + 4] = static_cast<scalar_t>(temp_boxes[i].z_min);
        h_bounding_boxes[i * 6 + 5] = static_cast<scalar_t>(temp_boxes[i].z_max);
    }
}