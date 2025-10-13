/**
 * @file kabsch.cc
 * @brief Kabsch coordinate system algorithms for baseline CPU implementation
 */

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
                                double &s1_len_out) {
    // Define the local Kabsch basis vectors:
    // e1 is perpendicular to the scattering plane
    Eigen::Vector3d e1 = s1_c.cross(s0).normalized();

    // e2 lies within the scattering plane, orthogonal to e1
    Eigen::Vector3d e2 = s1_c.cross(e1).normalized();

    // e3 bisects the angle between s0 and s1
    Eigen::Vector3d e3 = (s1_c + s0).normalized();

    // Compute the length of the predicted diffracted vector (|s₁|)
    double s1_len = s1_c.norm();
    s1_len_out = s1_len;

    // Rotation offset between the pixel and reflection centre
    double dphi = phi_pixel - phi_c;

    // Compute the predicted diffracted vector at φ′
    Eigen::Vector3d s1_phi_prime = s1_c + e3 * dphi;

    // Difference vector between pixel's s′ and the φ′-adjusted centroid
    Eigen::Vector3d deltaS = s_pixel - s1_phi_prime;

    // ε₁: displacement along e1, normalised by |s₁|
    double eps1 = e1.dot(deltaS) / s1_len;

    // ε₂: displacement along e2, with correction for non-orthogonality to e3
    double eps2 = e2.dot(deltaS) / s1_len - (e2.dot(e3) * dphi) / s1_len;

    // ε₃: displacement along rotation axis, scaled by ζ = m₂ · e₁
    double zeta = rot_axis.dot(e1);
    double eps3 = zeta * dphi;

    return {eps1, eps2, eps3};
}
