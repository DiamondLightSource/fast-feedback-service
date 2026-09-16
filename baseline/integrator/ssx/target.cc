#include "target.hpp"
#include "ellipsoid_parameterisation.hpp"
#include "calculations.hpp"

#include <algorithm>
#include <ranges>

std::vector<double>
MaximumLikelihoodTarget::damp_outlier_intensity_weights(
    const std::vector<double>& values)
{
    if (values.empty()) {
        return {};
    }

    auto damped = values;
    auto sorted = values;

    std::ranges::sort(sorted);

    const std::size_t n = sorted.size();
    const double q1 = sorted[n / 4];
    const double q3 = sorted[(3 * n) / 4];
    const double iqr = q3 - q1;
    const double threshold = q3 + 1.5 * iqr;

    for (auto& value : damped) {
        if (value > threshold) {
            value = threshold;
        }
    }

    return damped;
}

MaximumLikelihoodTarget::MaximumLikelihoodTarget(
    const Simple6MosaicityParameterisation& model,
    const Eigen::Matrix3d& A,
    const Eigen::Vector3d& s0,
    const std::vector<Eigen::Vector3d>& sp_list,
    const std::vector<Eigen::Vector3d>& covariances,
    const std::vector<double>& intensities,
    const std::vector<Eigen::Vector3i>& miller_indices,
    const std::vector<Eigen::Vector2d>& mobs)
    : model_(model)
{
    const std::size_t n = miller_indices.size();

    data_.reserve(n);

    std::vector<double> damped_intensities =
        damp_outlier_intensity_weights(intensities);

    for (std::size_t i = 0; i < n; ++i) {
        Eigen::Matrix2d sobs;

        sobs << covariances[i][0],
                covariances[i][2],
                covariances[i][2],
                covariances[i][1];

        data_.emplace_back(
            model_,
            A,
            s0,
            sp_list[i],
            miller_indices[i],
            damped_intensities[i],
            mobs[i],
            sobs);
    }
}

MaximumLikelihoodTarget::MaximumLikelihoodTarget(
    const Simple6MosaicityParameterisation& model,
    const Eigen::Matrix3d& A,
    const Eigen::Vector3d& s0,
    const std::vector<Eigen::Vector3d>& xyzobs_px,
    const std::vector<Eigen::Vector3d>& covariances,
    const std::vector<double>& intensities,
    const std::vector<Eigen::Vector3i>& miller_indices,
    const std::vector<Eigen::Vector2d>& mobs,
    const Panel& panel)
    : model_(model)
{
    const std::size_t n = miller_indices.size();

    const double s0_length = s0.norm();

    data_.reserve(n);

    std::vector<double> damped_intensities =
        damp_outlier_intensity_weights(intensities);

    for (std::size_t i = 0; i < n; ++i) {

        auto [xomm, yomm] =
            panel.px_to_mm(xyzobs_px[i][0], xyzobs_px[i][1]);

        Vector3d sp =
            panel.get_lab_coord(xomm, yomm);

        sp.normalize();
        sp *= s0_length;

        Eigen::Matrix2d sobs;

        sobs << covariances[i][0],
                covariances[i][2],
                covariances[i][2],
                covariances[i][1];

        data_.emplace_back(
            model_,
            A,
            s0,
            sp,
            miller_indices[i],
            damped_intensities[i],
            mobs[i],
            sobs);
    }
}

void MaximumLikelihoodTarget::update()
{
    for (auto& r : data_) {
        r.update();
    }
}

double MaximumLikelihoodTarget::log_likelihood() const
{
    double lnL = 0.0;

    for (const auto& r : data_) {
        lnL += r.log_likelihood();
    }
    return lnL;
}

double MaximumLikelihoodTarget::mse() const
{
    double mse = 0.0;

    for (const auto& r : data_) {

        Eigen::Vector2d diff =
            r.mobs() -
            r.conditional().mean();

        mse += diff.squaredNorm();
    }

    return mse / static_cast<double>(data_.size());
}

ParameterVector
MaximumLikelihoodTarget::first_derivatives() const
{
    ParameterVector dL =
        ParameterVector::Zero();

    for (const auto& r : data_) {
        dL += r.first_derivatives();
    }

    return dL;
}

FisherMatrix
MaximumLikelihoodTarget::fisher_information() const
{
    FisherMatrix I =
        FisherMatrix::Zero();

    for (const auto& r : data_) {
        I += r.fisher_information();
    }

    return I;
}