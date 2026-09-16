#pragma once

#include <Eigen/Core>
#include <array>

#include "mosaicity_parameterisation.hpp"

using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Matrix2d = Eigen::Matrix2d;
using Matrix3d = Eigen::Matrix3d;

using ParameterVector = Eigen::Matrix<double, 6, 1>;
using FisherMatrix = Eigen::Matrix<double, 6, 6>;
using DerivativeMatrices = std::array<Eigen::Matrix3d, 6>;
using DerivativeVectors = std::array<Vector3d, 6>;
using SigmaDerivativeMatrices = std::array<Matrix2d, 6>;
using MuDerivativeVectors = std::array<Vector2d, 6>;

Matrix2d compute_dSbar(const Matrix3d& S, const Matrix3d& dS);

Vector2d compute_dmbar(const Matrix3d& S, const Matrix3d& dS, double epsilon);

DerivativeMatrices rotate_Matrix3d_double(
    const Matrix3d& R,
    const DerivativeMatrices& A);

Matrix3d compute_change_of_basis_operation(const Vector3d& s0, const Vector3d& s2);

class ConditionalDistribution {
public:

    ConditionalDistribution(
        double norm_s0,
        const Vector3d& mu,
        const Matrix3d& S,
        const DerivativeMatrices& dS);

    const Vector2d& mean() const;
    const Matrix2d& sigma() const;
    double epsilon() const;

    const SigmaDerivativeMatrices&
    first_derivatives_of_sigma() const;

    const MuDerivativeVectors&
    first_derivatives_of_mean() const;

private:

    Vector3d mu_;
    Matrix3d S_;
    DerivativeMatrices dS_;

    double epsilon_;

    Vector2d mubar_;
    Matrix2d Sbar_;

    SigmaDerivativeMatrices dSbar_;
    MuDerivativeVectors dmbar_;
};

class ReflectionLikelihood {
public:

    ReflectionLikelihood(
        const Simple6MosaicityParameterisation& model,
        const Matrix3d& A,
        const Vector3d& s0,
        const Vector3d& sp,
        const Eigen::Vector3i& h,
        double ctot,
        const Vector2d& mobs,
        const Matrix2d& sobs);

    void update();

    double log_likelihood() const;

    ParameterVector first_derivatives() const;

    FisherMatrix fisher_information() const;

    const ConditionalDistribution& conditional() const;

    const Vector2d& mobs() const;

private:

    const Simple6MosaicityParameterisation& model_;

    Vector3d s0_;
    Vector3d sp_;
    Vector3d r_;

    double norm_s0_;
    double ctot_;

    Matrix2d sobs_;

    Matrix3d R_;
    Vector2d mobs_;

    Vector3d mu_;
    Matrix3d S_;

    DerivativeMatrices dS_;

    ConditionalDistribution conditional_;
};
