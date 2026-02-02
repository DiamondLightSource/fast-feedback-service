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
 */

#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_common.hpp"
#include "extent.hpp"
#include "integrator.cuh"
#include "kabsch.cuh"
#include "math/math_utils.cuh"

using namespace fastvec;

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
 * @brief CUDA kernel to compute pixel-to-Kabsch transformations for an image
 * 
 * This kernel transforms pixel coordinates from an image into Kabsch coordinates.
 * Each thread processes one pixel, checking if it falls within any reflection's
 * bounding box and computing Kabsch coordinates if so.
 * 
 * @param d_image Pointer to image data
 * @param image_pitch Pitch of the image data in bytes
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param image_num Current image number in the sequence
 * @param d_matrix Detector coordinate transformation matrix (3x3)
 * @param wavelength X-ray wavelength in Angstroms
 * @param osc_start Starting oscillation angle in radians
 * @param osc_width Oscillation width per image in radians
 * @param image_range_start First image number in the range
 * @param s0 Incident beam vector
 * @param rot_axis Rotation axis vector
 * @param d_s1_vectors Array of s1 vectors for each reflection
 * @param d_phi_values Array of phi values for each reflection
 * @param d_bboxes Array of bounding box structs for each reflection
 * @param d_reflection_indices Indices of reflections touching this image
 * @param num_reflections_this_image Number of reflections to process
 */
__global__ void kabsch_transform(const void *d_image,
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
                                 size_t num_reflections_this_image) {
    // Calculate global thread index
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds guard
    if (x >= width + 1 || y >= height + 1) return;

    // First pass: find all reflections whose bbox contains this pixel
    // Use a small local array - typically very few reflections overlap at any pixel
    constexpr size_t MAX_OVERLAPPING_REFLECTIONS = 16;
    size_t matching_refl_indices[MAX_OVERLAPPING_REFLECTIONS];
    size_t num_matches = 0;

    for (size_t i = 0; i < num_reflections_this_image; ++i) {
        const size_t refl_idx = d_reflection_indices[i];
        const BoundingBoxExtents &bbox = d_bboxes[refl_idx];

        // Check if pixel is inside this reflection's bounding box
        // Bbox is half-open: [min, max), but we extend upper bounds by 1
        // because pixel (x, y) contributes to corner (x - 0.5, y - 0.5)
        // and we need to include corners at the far edges of the bbox
        const bool inside_bbox = (x >= bbox.x_min && x <= bbox.x_max)
                                 && (y >= bbox.y_min && y <= bbox.y_max)
                                 && (image_num >= bbox.z_min && image_num < bbox.z_max);

        if (inside_bbox) {
            if (num_matches < MAX_OVERLAPPING_REFLECTIONS) {
                matching_refl_indices[num_matches++] = refl_idx;
            }
            // If we exceed MAX_OVERLAPPING_REFLECTIONS, silently ignore extras
            // This is a rare edge case - could add error handling if needed
        }
    }

    // Exit early if pixel is not within any reflection's bounding box
    if (num_matches == 0) return;

    // Shift to corner coordinates
    // Pixel centre are at integer (x, y), corners at (x - 0.5, y - 0.5)
    const scalar_t x_corner = static_cast<scalar_t>(x) - scalar_t(0.5);
    const scalar_t y_corner = static_cast<scalar_t>(y) - scalar_t(0.5);

    // Second pass: process only the matching reflections
    for (size_t i = 0; i < num_matches; ++i) {
        const size_t refl_idx = matching_refl_indices[i];

        // Get reflection data
        const Vector3D &s1_c = d_s1_vectors[refl_idx];
        const scalar_t phi_c = d_phi_values[refl_idx];

        // Compute s_pixel from corner coordinates
        // Transform pixel corner to lab coordinates using detector matrix
        // lab_coord = d_matrix * (x_corner, y_corner, 1)
        Vector3D lab_coord =
          make_vector3d(d_matrix[0] * x_corner + d_matrix[1] * y_corner + d_matrix[2],
                        d_matrix[3] * x_corner + d_matrix[4] * y_corner + d_matrix[5],
                        d_matrix[6] * x_corner + d_matrix[7] * y_corner + d_matrix[8]);
        // Normalize to get s_pixel and scale by 1/wavelength
        Vector3D s_pixel = normalized(lab_coord) / wavelength;

        // Compute phi_pixel from image_num
        scalar_t phi_pixel =
          osc_start
          + (static_cast<scalar_t>(image_num - image_range_start) + scalar_t(0.5))
              * osc_width;

        // Compute Kabsch transformation
        scalar_t s1_len;
        Vector3D eps =
          pixel_to_kabsch(s0, s1_c, phi_c, s_pixel, phi_pixel, rot_axis, s1_len);

        // Store results. Output arrays?
    }
}

/**
 * @brief Host wrapper function for image-based Kabsch computation
 * 
 * Launches a kernel over the entire image grid (width+1 x height+1 for corners).
 * Each thread checks if its pixel falls within any reflection's bounding box,
 * and if so, computes the Kabsch transform.
 */
void compute_kabsch_transform(const void *d_image,
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
                              cudaStream_t stream) {
    // Configure kernel launch parameters

    /*
     * Wide but small blocks tend to work well for 2D image processing
     * as they provide good memory coalescing along rows.
     * 
     * Alternatively, could experiment with square blocks (16x16)
     * for more balanced workload distribution and better cache usage
     * and occupancy.
    */

    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    dim3 blockDim(BLOCK_X, BLOCK_Y);
    // Launch grid covers entire image + 1 for corner sampling
    dim3 gridDim(ceil_div(width + 1u, blockDim.x), ceil_div(height + 1u, blockDim.y));

    // TODO: Launch the kernel
    // kabsch_transform_image<<<gridDim, blockDim, 0, stream>>>(
    //     d_image, image_pitch, width, height, image_num,
    //     d_matrix, wavelength, osc_start, osc_width, image_range_start,
    //     s0, rot_axis, d_s1_vectors, d_phi_values, d_bboxes,
    //     d_reflection_indices, num_reflections_this_image);

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