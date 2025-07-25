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

#include "math/vector3d.cuh"

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
void compute_voxel_kabsch(const fastfb::Vector3D* h_s_pixels,
                          const double* h_phi_pixels,
                          fastfb::Vector3D s1_c,
                          double phi_c,
                          fastfb::Vector3D s0,
                          fastfb::Vector3D rot_axis,
                          fastfb::Vector3D* h_eps,
                          double* h_s1_len,
                          size_t n);