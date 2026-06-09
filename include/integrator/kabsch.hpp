/**
 * @file kabsch.hpp
 * @brief Kabsch coordinate system algorithms for baseline CPU integration.
 */

#pragma once

#include <Eigen/Dense>

/**
 * @brief Transform a pixel from reciprocal space into the local Kabsch
 * coordinate frame.
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
 * @param s1_c Predicted diffracted vector at the reflection centre
 * (s₁ᶜ), units of 1/Å
 * @param phi_c Rotation angle at the reflection centre (φᶜ), in radians
 * @param s_pixel Diffracted vector at the current pixel (S′), units of
 * 1/Å
 * @param phi_pixel Rotation angle at the pixel (φ′), in radians
 * @param rot_axis Unit goniometer rotation axis vector (m₂)
 * @param s1_len_out Optional output for magnitude of s₁ᶜ (|s₁|)
 * @return Eigen::Vector3d The local coordinates (ε₁, ε₂, ε₃) in Kabsch
 * space
 */
Eigen::Vector3d pixel_to_kabsch(const Eigen::Vector3d &s0,
                                const Eigen::Vector3d &s1_c,
                                double phi_c,
                                const Eigen::Vector3d &s_pixel,
                                double phi_pixel,
                                const Eigen::Vector3d &rot_axis,
                                double &s1_len_out);
