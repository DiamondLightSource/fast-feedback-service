#pragma once

#include <dx2/detector.hpp>
#include "calculations.hpp"

class MaximumLikelihoodTarget {
public:

    using ReflectionList = std::vector<ReflectionLikelihood>;

    MaximumLikelihoodTarget(
        const Simple6MosaicityParameterisation& model,
        const Eigen::Matrix3d& A,
        const Eigen::Vector3d& s0,
        const std::vector<Eigen::Vector3d>& xyzobs_px,
        const std::vector<Eigen::Vector3d>& covariances,
        const std::vector<double>& intensities,
        const std::vector<Eigen::Vector3i>& miller_indices,
        const std::vector<Eigen::Vector2d>& mobs,
        const Panel& panel);

    MaximumLikelihoodTarget(
        const Simple6MosaicityParameterisation& model,
        const Eigen::Matrix3d& A,
        const Eigen::Vector3d& s0,
        const std::vector<Eigen::Vector3d>& sp_list,
        const std::vector<Eigen::Vector3d>& covariances,
        const std::vector<double>& intensities,
        const std::vector<Eigen::Vector3i>& miller_indices,
        const std::vector<Eigen::Vector2d>& mobs);

    void update();

    double mse() const;

    double log_likelihood() const;

    ParameterVector first_derivatives() const;

    FisherMatrix fisher_information() const;

    const ReflectionList& reflections() const {
        return data_;
    }

private:

    const Simple6MosaicityParameterisation& model_;

    ReflectionList data_;

    std::vector<double> damp_outlier_intensity_weights(
        const std::vector<double>& values);
};

