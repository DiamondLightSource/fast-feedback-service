#include "reflection_likelihood.hpp"

#include <cmath>
#include <Eigen/Dense>

/*
This is the per-reflection calculation code.
Using the observed covariance, observed centroid and the 6-parameter mosaicity model,
this calculates the Likelihood, derivatives and fisher information for the refinement.
*/

Matrix2d compute_dSbar(
    const Matrix3d& S,
    const Matrix3d& dS)
{
    // how to efficiently assert that abs(S(2,2)) > eps during refinement cycles?
    Eigen::Vector2d S12 = S.block<2,1>(0,2);
    Eigen::RowVector2d S21 = S.block<1,2>(2,0);
    double S22_inv = 1.0 / S(2,2);

    Matrix2d B = (S12 * (S22_inv * dS(2,2) * S22_inv)) * S21;
    Matrix2d C = (S12 * S22_inv) * dS.block<1,2>(2,0);
    Matrix2d D = (dS.block<2,1>(0,2) * S22_inv) * S21;

    return dS.block<2,2>(0,0) + B - (C + D);
}

Vector2d compute_dmbar(
    const Matrix3d& S,
    const Matrix3d& dS,
    double epsilon)
{
    // how to efficiently assert that abs(S(2,2)) > eps during refinement cycles?
    double S22_inv = 1.0 / S(2,2);
    Vector2d B = dS.block<2,1>(0,2) * (S22_inv * epsilon);
    Vector2d C = -S.block<2,1>(0,2) * (S22_inv * dS(2,2) * S22_inv * epsilon);

    return B + C;
}

DerivativeMatrices rotate_mat3_double(
    const Eigen::Matrix3d& R,
    const DerivativeMatrices& A)
{
    DerivativeMatrices result;
    for (std::size_t i = 0; i < 6; ++i) {
        result[i] = R * A[i] * R.transpose();
    }
    return result;
}

Eigen::Matrix3d compute_change_of_basis_operation(
    const Eigen::Vector3d& s0,
    const Eigen::Vector3d& s2)
{
    // add check that s0 and s2 are not nearly parallel?
    const Eigen::Vector3d e1 = s2.cross(s0).normalized();
    const Eigen::Vector3d e2 = s2.cross(e1).normalized();
    const Eigen::Vector3d e3 = s2.normalized();

    Eigen::Matrix3d R;
    R <<
        e1.x(), e1.y(), e1.z(),
        e2.x(), e2.y(), e2.z(),
        e3.x(), e3.y(), e3.z();
    return R;
}

namespace {

Eigen::Matrix3d compute_S(
    const Simple6MosaicityParameterisation& model,
    const Eigen::Matrix3d& R)
{
    return R * model.sigma() * R.transpose();
}

DerivativeMatrices compute_dS(
    const Simple6MosaicityParameterisation& model,
    const Eigen::Matrix3d& R)
{
    return rotate_mat3_double(R, model.first_derivatives());
}

} // namespace

ConditionalDistribution::ConditionalDistribution(
    double norm_s0,
    const Vector3d& mu,
    const Matrix3d& S,
    const DerivativeMatrices& dS)
    : mu_(mu),
      S_(S),
      dS_(dS)
{
    Matrix2d S11 = S.block<2,2>(0,0);
    Eigen::Vector2d S12 = S.block<2,1>(0,2);
    Eigen::RowVector2d S21 = S.block<1,2>(2,0);

    double S22 = S(2,2);
    Vector2d mu1 = mu.head<2>();
    double mu2 = mu(2);

    epsilon_ = norm_s0 - mu2;
    mubar_ = mu1 + S12 * (epsilon_ / S22);
    Sbar_ = S11 - (S12 / S22) * S21;

    for (std::size_t i = 0; i < dSbar_.size(); ++i){
        dSbar_[i] = compute_dSbar(S_, dS_[i]);
    }
    for (std::size_t i = 0; i < dmbar_.size(); ++i){
        dmbar_[i] = compute_dmbar(S_, dS_[i], epsilon_);
    }
}

const Vector2d& ConditionalDistribution::mean() const {
    return mubar_;
}

const Matrix2d& ConditionalDistribution::sigma() const {
    return Sbar_;
}

double ConditionalDistribution::epsilon() const {
    return epsilon_;
}

const SigmaDerivativeMatrices&
ConditionalDistribution::first_derivatives_of_sigma() const {
    return dSbar_;
}

const MuDerivativeVectors&
ConditionalDistribution::first_derivatives_of_mean() const {
    return dmbar_;
}

ReflectionLikelihood::ReflectionLikelihood(
    const Simple6MosaicityParameterisation& model,
    const Eigen::Matrix3d& A,
    const Eigen::Vector3d& s0,
    const Eigen::Vector3d& sp,
    const Eigen::Vector3i& h,
    double ctot,
    const Eigen::Vector2d& mobs,
    const Eigen::Matrix2d& sobs)
    :
    model_(model),
    s0_(s0),
    sp_(sp),
    r_(A * h.cast<double>()),
    norm_s0_(s0.norm()),
    ctot_(ctot),
    sobs_(sobs),
    R_(compute_change_of_basis_operation(s0, sp)),
    mobs_(mobs),
    mu_(R_ * (s0_ + r_)),
    S_(compute_S(model_, R_)),
    dS_(compute_dS(model_, R_)),
    conditional_(norm_s0_, mu_, S_, dS_) {}

const ConditionalDistribution& ReflectionLikelihood::conditional() const {
    return conditional_;
}
const Vector2d& ReflectionLikelihood::mobs() const {
    return mobs_;
}


void ReflectionLikelihood::update(){
    // s2 unchanged as r unchanged
    // mu unchanged, so no dmu term.
    S_ = compute_S(model_, R_);
    dS_ = compute_dS(model_, R_);
    // make a new conditional distribution with updated values.
    conditional_ = ConditionalDistribution(norm_s0_, mu_, S_, dS_);
}

double ReflectionLikelihood::log_likelihood() const {
    // Calculates the log likelihood.

    // Marginal
    double S22 = S_(2, 2);
    const double S22_inv = 1.0 / S22;
    double mu2 = mu_(2);

    // Conditional
    const Matrix2d& Sbar = conditional_.sigma();
    Vector2d mubar = conditional_.mean();
    Matrix2d Sbar_inv = Sbar.inverse();
    double Sbar_det = Sbar.determinant();

    // Use ctot as weights for marginal and conditional components

    // Compute the marginal likelihood
    double m_d = norm_s0_ - mu2;
    double m_lnL = ctot_ * (std::log(S22) + S22_inv * std::pow(m_d,2));

    //Compute the conditional likelihood
    Vector2d c_d = mobs_ - mubar;
    Eigen::Matrix2d V = sobs_ + c_d * c_d.transpose();
    double c_lnL = ctot_ * (std::log(Sbar_det) + (Sbar_inv * V).trace());

    //Return the joint likelihood
    double jLL = -0.5 * (m_lnL + c_lnL);
    return jLL;
}

ParameterVector ReflectionLikelihood::first_derivatives() const {

    // Marginal
    const double S22 = S_(2, 2);
    const double S22_inv = 1.0 / S22;
    const double mu2 = mu_(2);

    // Conditional
    const Matrix2d& Sbar = conditional_.sigma();
    const Vector2d& mubar = conditional_.mean();
    const SigmaDerivativeMatrices& dSbar = conditional_.first_derivatives_of_sigma();
    const MuDerivativeVectors& dmbar = conditional_.first_derivatives_of_mean();
    const Matrix2d Sbar_inv = Sbar.inverse();

    const double epsilon = norm_s0_ - mu2;
    const Vector2d c_d = mobs_ - mubar;

    // Use ctot as weights for marginal and conditional components
    // Precalculate a few things.
    const Matrix2d I = Matrix2d::Identity();
    const Matrix2d V1 = sobs_ + c_d * c_d.transpose();
    const Matrix2d V2 = I - Sbar_inv * V1;
    const Vector2d Sbar_inv_cd = Sbar_inv * c_d;

    ParameterVector U_vec = ParameterVector::Zero();
    ParameterVector V_vec = ParameterVector::Zero();
    ParameterVector W_vec = ParameterVector::Zero();

    for (std::size_t i = 0; i < 6; ++i) {
        U_vec(i) = ctot_ * S22_inv * dS_[i](2,2) * (1.0 - S22_inv * epsilon * epsilon);
        V_vec(i) = ctot_ * (Sbar_inv * dSbar[i] * V2).trace();
        //W_vec(i) = -2.0 * ctot_ * (Sbar_inv * (c_d * dmbar[i].transpose())).trace();
        W_vec(i) = -2.0 * ctot_ * dmbar[i].dot(Sbar_inv_cd); // Using tr(AuvT) = vTau
    }
    return -0.5 * (U_vec + V_vec + W_vec);
}

FisherMatrix ReflectionLikelihood::fisher_information() const {

    const double S22 = S_(2, 2);
    const double S22_inv = 1.0 / S22;
    const Matrix2d& Sbar = conditional_.sigma();
    const SigmaDerivativeMatrices& dSbar = conditional_.first_derivatives_of_sigma();
    const MuDerivativeVectors& dmbar = conditional_.first_derivatives_of_mean();
    const Matrix2d Sbar_inv = Sbar.inverse();
    FisherMatrix I = FisherMatrix::Zero();

    for (std::size_t j = 0; j < 6; ++j) {
        const double dS22_j = dS_[j](2, 2);
        for (std::size_t i = 0; i < 6; ++i) {
            const double dS22_i = dS_[i](2, 2);
            // Marginal contribution
            const double U = S22_inv * dS22_j * S22_inv *dS22_i;
            // Conditional covariance contribution
            const double V = (Sbar_inv * dSbar[j] * Sbar_inv * dSbar[i]).trace();
            // Conditional mean contribution
            const double W = 2.0 * (Sbar_inv * dmbar[i] * dmbar[j].transpose()).trace();
            I(j, i) = 0.5 * ctot_ * (V + W + U);
        }
    }
    return I;
}