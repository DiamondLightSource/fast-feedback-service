/**
 * @file sigma_estimation.hpp
 * @brief Estimation of spot extent parameters (σ_b, σ_m) for baseline CPU
 *        integration.
 */

#pragma once

#include <Eigen/Dense>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/reflection.hpp>
#include <utility>

/**
 * @brief Calculate the squared deviation in Kabsch space between the predicted
 * and observed reflection positions.
 *
 * @param xyzcal Predicted centroid (mm).
 * @param xyzobs Observed centroid (mm).
 * @param s0 Incident beam vector.
 * @param panel Detector panel for lab-coordinate conversion.
 * @param m2 Goniometer rotation axis.
 * @return {variance in (ε₁,ε₂) plane, variance along ε₃}
 */
std::pair<double, double> squaredev_in_kabsch_space(const Eigen::Vector3d &xyzcal,
                                                    const Eigen::Vector3d &xyzobs,
                                                    const Eigen::Vector3d &s0,
                                                    const Panel &panel,
                                                    Eigen::Vector3d m2);

/**
 * @brief Estimate the total spot extent parameters (σ_b, σ_m) from a set of
 * indexed/refined reflections.
 *
 * @param indexed Reflection table containing the per-reflection variances and
 *                predicted/observed centroids.
 * @param expt Experiment model providing beam and goniometer geometry.
 * @param min_bbox_depth Minimum bounding-box depth (in images) for a reflection
 *                       to contribute to the σ_m estimate.
 * @return {total σ_b, total σ_m} in radians.
 */
std::pair<double, double> estimate_sigmas(ReflectionTable const &indexed,
                                          Experiment &expt,
                                          int min_bbox_depth = 6);
