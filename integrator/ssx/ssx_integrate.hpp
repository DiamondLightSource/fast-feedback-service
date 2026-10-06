#pragma once
#include <Eigen/Dense>
#include <dx2/detector.hpp>
#include <vector>
#include "predictor/predict.hpp"
#include "mosaicity_parameterisation.hpp"

using Vector3d = Eigen::Vector3d;
using Vector2d = Eigen::Vector2d;

void ssx_integrate(const std::vector<Vector3d> &xyzcal_px,
                       const std::vector<Vector3d> &xyzobs_px,
                       const std::vector<Vector3d> &covariances,
                       const std::vector<double> &intensities,
                       const std::vector<Eigen::Vector3i> &miller_indices,
                       const std::vector<Vector2d> &mobs,
                       const Vector3d &s0,
                       const Panel &panel,
                       const Matrix3d &A);

Simple6MosaicityParameterisation refine_mosaicity(const std::vector<Eigen::Vector3d> &xyzcal_px,
                       const std::vector<Eigen::Vector3d> &xyzobs_px,
                       const std::vector<Eigen::Vector3d> &covariances,
                       const std::vector<double> &intensities,
                       const std::vector<Eigen::Vector3i> &miller_indices,
                       const std::vector<Eigen::Vector2d> &mobs,
                       const Eigen::Vector3d &s0,
                       const Panel &panel,
                       const Eigen::Matrix3d &A);

std::vector<Prediction> predict_ssx(
    const Simple6MosaicityParameterisation& model,
    const Vector3d &s0,
    const Panel &panel,
    const Matrix3d &A);
