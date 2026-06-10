/**
 * @file ray_predictors.hpp
 * @brief Ray-prediction routines for the various scattering geometries.
 */

#pragma once

#include <Eigen/Dense>
#include <array>
#include <optional>

#include "predictor/utils.hpp"

/**
 * @brief Return a Ray object if a prediction is found within a given tolerance
 * angle (monochromatic stills).
 */
std::optional<Ray> predict_ray_monochromatic_stills(const std::array<int, 3> &index,
                                                    const Eigen::Matrix3d &A,
                                                    const Eigen::Vector3d &s0,
                                                    const double dmin,
                                                    const double delta_psi_tolerance);

/**
 * @brief Return up to two Ray objects if a prediction is found during a static
 * rotation (monochromatic).
 */
std::array<std::optional<Ray>, 2> predict_ray_monochromatic_static(
  const std::array<int, 3> &index,
  const Eigen::Matrix3d &A,
  const Eigen::Matrix3d &r_setting,
  const Eigen::Matrix3d &r_setting_inv,
  const Eigen::Vector3d &s0,
  const Eigen::Vector3d &m2,
  const Rotator &rotator,
  const double dmin,
  const double phi_beg,
  const double d_osc);

/**
 * @brief Return a Ray object if a prediction is found during a scan-varying
 * rotation (monochromatic).
 */
std::optional<Ray> predict_ray_monochromatic_sv(const std::array<int, 3> &index,
                                                const Eigen::Matrix3d &A1,
                                                const Eigen::Matrix3d &A2,
                                                const Eigen::Vector3d &s0_1,
                                                const Eigen::Vector3d &s0_2,
                                                const double dmin,
                                                const double phi_beg,
                                                const double d_osc);

/**
 * @brief Return a Ray object if a prediction is found for a polychromatic beam
 * (stills).
 */
std::optional<Ray> predict_ray_polychromatic_stills(const std::array<int, 3> &index,
                                                    const Eigen::Matrix3d &A,
                                                    Eigen::Vector3d s0_unit,
                                                    const double wavelength_min,
                                                    const double wavelength_max,
                                                    const double dmin);

/**
 * @brief Return a Ray object if a prediction is found during a scan-varying
 * rotation for a polychromatic beam.
 */
std::optional<Ray> predict_ray_polychromatic_rotational(const std::array<int, 3> &index,
                                                        const Eigen::Matrix3d &A1,
                                                        const Eigen::Matrix3d &A2,
                                                        Eigen::Vector3d s0_1_unit,
                                                        Eigen::Vector3d s0_2_unit,
                                                        const double wavelength_min,
                                                        const double wavelength_max,
                                                        const double dmin,
                                                        const double phi_beg,
                                                        const double d_osc);
