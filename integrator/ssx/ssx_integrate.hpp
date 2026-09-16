#pragma once
#include <vector>
#include <Eigen/Dense>
#include <dx2/detector.hpp>

using Vector3d = Eigen::Vector3d;

Vector3d ssx_integrate(const std::vector<Eigen::Vector3d>& xyzcal_px,
    const std::vector<Eigen::Vector3d>& xyzobs_px,
    const std::vector<Eigen::Vector3d>& covariances,
    const std::vector<double>& intensities,
    const std::vector<Eigen::Vector3i>& miller_indices,
    const std::vector<Eigen::Vector2d>& mobs,
    const Eigen::Vector3d &s0,
    const Panel &panel,
    const Eigen::Matrix3d& A
);