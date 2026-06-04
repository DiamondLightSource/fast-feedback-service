/**
 * @file coordinate_system.hpp
 * @brief Kabsch reciprocal-space coordinate system for a reflection
 */

#pragma once

#include <Eigen/Dense>

/**
 * @brief Per-reflection Kabsch coordinate system.
 *
 * Builds the local orthonormal basis (e1, e2) around the diffracted beam
 * vector s₁ and provides the rotation/zeta factor used to map detector
 * observations into the reflection profile frame.
 */
class CoordinateSystem {
  public:
    /**
     * @brief Build the Kabsch coordinate system for a single reflection.
     *
     * @param m2 Goniometer rotation axis (m₂)
     * @param s0 Incident beam vector (s₀)
     * @param s1 Diffracted beam vector (s₁) at the reflecting position
     * @param phi Rotation angle φ at which the reflection is in diffracting
     *            condition, in radians
     */
    CoordinateSystem(Eigen::Vector3d m2,
                     Eigen::Vector3d s0,
                     Eigen::Vector3d s1,
                     double phi);

    /**
     * @brief Map a diffracted beam vector into this Kabsch coordinate frame.
     *
     * @param s_dash Diffracted beam vector (s') to project
     * @param phi_dash Rotation angle φ' associated with @p s_dash, in radians
     * @return Coordinates (e1, e2, rotation) in the reflection profile frame
     */
    Eigen::Vector3d coords_from_s1vector(const Eigen::Vector3d s_dash,
                                         double phi_dash) const;

    /**
     * @brief Rotation efficiency factor ζ.
     *
     * @return Projection of the (normalized) rotation axis onto e1
     */
    double zeta() const;

  private:
    Eigen::Vector3d s1_;
    double phi_;
    double zeta_;
    Eigen::Vector3d scaled_e1_;
    Eigen::Vector3d scaled_e2_;
};
