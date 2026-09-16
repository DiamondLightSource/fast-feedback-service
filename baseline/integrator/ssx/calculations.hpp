#pragma once

#include <Eigen/Core>
#include <array>

#include "ellipsoid_parameterisation.hpp"

using Matrix3d = Eigen::Matrix3d;
using Matrix2d = Eigen::Matrix2d;
using Vector2d = Eigen::Vector2d;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat2 = Eigen::Matrix2d;
using Mat3 = Eigen::Matrix3d;

using ParameterVector = Eigen::Matrix<double, 6, 1>;
using FisherMatrix = Eigen::Matrix<double, 6, 6>;
using DerivativeMatrices = std::array<Eigen::Matrix3d, 6>;
using DerivativeVectors = std::array<Vec3, 6>;
using SigmaDerivativeMatrices = std::array<Eigen::Matrix2d, 6>;
using MuDerivativeVectors = std::array<Eigen::Vector2d, 6>;

Matrix2d compute_dSbar(
    const Matrix3d& S,
    const Matrix3d& dS);

Vector2d compute_dmbar(
    const Matrix3d& S,
    const Matrix3d& dS,
    double epsilon);

DerivativeMatrices rotate_mat3_double(
    const Eigen::Matrix3d& R,
    const DerivativeMatrices& A);

Eigen::Matrix3d compute_change_of_basis_operation(
    const Eigen::Vector3d& s0,
    const Eigen::Vector3d& s2);

class ConditionalDistribution {
public:

    ConditionalDistribution(
        double norm_s0,
        const Vec3& mu,
        const Mat3& S,
        const DerivativeMatrices& dS);

    const Vec2& mean() const;
    const Mat2& sigma() const;
    double epsilon() const;

    const SigmaDerivativeMatrices&
    first_derivatives_of_sigma() const;

    const MuDerivativeVectors&
    first_derivatives_of_mean() const;

private:

    Vec3 mu_;
    Mat3 S_;
    DerivativeMatrices dS_;

    double epsilon_;

    Vec2 mubar_;
    Mat2 Sbar_;

    SigmaDerivativeMatrices dSbar_;
    MuDerivativeVectors dmbar_;
};

class ReflectionLikelihood {
public:

    ReflectionLikelihood(
        const Simple6MosaicityParameterisation& model,
        const Eigen::Matrix3d& A,
        const Eigen::Vector3d& s0,
        const Eigen::Vector3d& sp,
        const Eigen::Vector3i& h,
        double ctot,
        const Eigen::Vector2d& mobs,
        const Eigen::Matrix2d& sobs);

    void update();

    double log_likelihood() const;

    ParameterVector first_derivatives() const;

    FisherMatrix fisher_information() const;

    const ConditionalDistribution& conditional() const;

    const Vector2d& mobs() const;

private:

    const Simple6MosaicityParameterisation& model_;

    Eigen::Vector3d s0_;
    Eigen::Vector3d sp_;
    Eigen::Vector3d r_;

    double norm_s0_;
    double ctot_;

    Eigen::Matrix2d sobs_;

    Eigen::Matrix3d R_;
    Eigen::Vector2d mobs_;

    Eigen::Vector3d mu_;
    Eigen::Matrix3d S_;

    DerivativeMatrices dS_;

    ConditionalDistribution conditional_;
};

/*ParameterVector S;
FisherMatrix I;
ParameterVector p = I.ldlt().solve(S);*/