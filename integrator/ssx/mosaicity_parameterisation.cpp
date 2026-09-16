#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include "mosaicity_parameterisation.hpp"

Simple6MosaicityParameterisation::Simple6MosaicityParameterisation()
    : parameters_(Vector6d::Zero()) {}

Simple6MosaicityParameterisation::Simple6MosaicityParameterisation(
    const Vector6d& params)
    : parameters_(params) {}

Simple6MosaicityParameterisation Simple6MosaicityParameterisation::from_sigma_d(double sigma_d){
    // Isotropic mosaicity: M = sigma_d * I
    Vector6d p;
    p << sigma_d, 0.0, sigma_d, 0.0, 0.0, sigma_d;
    return Simple6MosaicityParameterisation(p);
}

const Vector6d& Simple6MosaicityParameterisation::parameters() const {
    return parameters_;
}

void Simple6MosaicityParameterisation::set_parameters(const Vector6d& p) {
    parameters_ = p;
}

Matrix3d Simple6MosaicityParameterisation::M() const {
    const auto& p = parameters_;
    Matrix3d M;
    M <<
        p(0), 0.0, 0.0,
        p(1), p(2), 0.0,
        p(3), p(4), p(5);
    return M;
}

Matrix3d Simple6MosaicityParameterisation::sigma() const {
    const Matrix3d m = M();
    return m * m.transpose();
}

DerivativeMatrices Simple6MosaicityParameterisation::first_derivatives() const {

    const double b1 = parameters_(0);
    const double b2 = parameters_(1);
    const double b3 = parameters_(2);
    const double b4 = parameters_(3);
    const double b5 = parameters_(4);
    const double b6 = parameters_(5);

    DerivativeMatrices dSigma;

    dSigma[0] <<
        2.0*b1, b2,     b4,
        b2,     0.0,    0.0,
        b4,     0.0,    0.0;

    dSigma[1] <<
        0.0,    b1,     0.0,
        b1,     2.0*b2, b4,
        0.0,    b4,     0.0;

    dSigma[2] <<
        0.0,    0.0,    0.0,
        0.0,    2.0*b3, b5,
        0.0,    b5,     0.0;

    dSigma[3] <<
        0.0,    0.0,    b1,
        0.0,    0.0,    b2,
        b1,     b2,     2.0*b4;

    dSigma[4] <<
        0.0,    0.0,    0.0,
        0.0,    0.0,    b3,
        0.0,    b3,     2.0*b5;

    dSigma[5] <<
        0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,
        0.0,    0.0,    2.0*b6;

    return dSigma;
}


Mosaicity Simple6MosaicityParameterisation::mosaicity() const {

    Eigen::SelfAdjointEigenSolver<Matrix3d> solver(sigma());

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Failed to compute eigendecomposition");
    }

    Eigen::Vector3d eig =
        solver.eigenvalues();

    double m1 = std::sqrt(std::max(0.0, eig(0)));
    double m2 = std::sqrt(std::max(0.0, eig(1)));
    double m3 = std::sqrt(std::max(0.0, eig(2)));

    return {m1, m2, m3};
}

void Simple6MosaicityParameterisation::print_mosaicity() const {
    Mosaicity m = mosaicity();
    std::cout
        << "\nInvariant crystal mosaicity:\n"
        << "M1 : " << m.min*1e6 << " muA^-1\n"
        << "M2 : " << m.mid*1e6 << " muA^-1\n"
        << "M3 : " << m.max*1e6 << " muA^-1\n";
}
