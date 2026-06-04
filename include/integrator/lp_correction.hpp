/**
 * @file lp_correction.hpp
 * @brief Lorentz and polarization correction calculations
 */

#pragma once

#include <Eigen/Dense>

/**
 * @brief Lorentz correction factor for a reflection.
 *
 * @param s0 Incident beam vector (s₀)
 * @param m2 Goniometer rotation axis (m₂)
 * @param s1 Diffracted beam vector (s₁)
 * @return Lorentz correction factor
 */
double lorentz_correction(const Eigen::Vector3d &s0,
                          const Eigen::Vector3d &m2,
                          const Eigen::Vector3d &s1);

/**
 * @brief Polarization correction factor for a reflection.
 *
 * @param s0 Incident beam vector (s₀)
 * @param pn Polarization normal
 * @param pf Polarization fraction
 * @param s1 Diffracted beam vector (s₁)
 * @return Polarization correction factor
 */
double polarization_correction(const Eigen::Vector3d &s0,
                               const Eigen::Vector3d &pn,
                               double pf,
                               const Eigen::Vector3d &s1);

/**
 * @brief Combined Lorentz-polarization correction for reflections sharing the
 * same beam geometry.
 *
 * Caches the fixed beam/goniometer geometry so the LP factor can be evaluated
 * per reflection from its diffracted beam vector alone.
 */
class LPCorrection {
  public:
    /**
     * @brief Construct an LP corrector for a fixed beam geometry.
     *
     * @param s0 Incident beam vector (s₀)
     * @param pn Polarization normal
     * @param pf Polarization fraction
     * @param m2 Goniometer rotation axis (m₂)
     */
    LPCorrection(Eigen::Vector3d s0, Eigen::Vector3d pn, double pf, Eigen::Vector3d m2);

    /**
     * @brief Compute the LP correction L/P for a diffracted beam vector.
     *
     * @param s1 Diffracted beam vector (s₁)
     * @return Combined Lorentz-polarization correction factor (L / P)
     */
    double calculate(const Eigen::Vector3d s1);

  private:
    Eigen::Vector3d s0_;
    Eigen::Vector3d pn_;
    double pf_;
    Eigen::Vector3d m2_;
};
