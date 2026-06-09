/**
 * @file kabsch.cc
 * @brief Kabsch coordinate system algorithms for baseline CPU integration.
 */

#include "integrator/kabsch.hpp"

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
