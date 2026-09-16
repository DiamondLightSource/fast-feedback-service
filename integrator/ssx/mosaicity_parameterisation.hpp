#pragma once

#include <Eigen/Core>
#include <array>

using DerivativeMatrices = std::array<Eigen::Matrix3d, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix3d = Eigen::Matrix3d;

struct Mosaicity {
    double min;
    double mid;
    double max;
};

class Simple6MosaicityParameterisation {
  public:
    Simple6MosaicityParameterisation();

    explicit Simple6MosaicityParameterisation(const Vector6d &params);

    static Simple6MosaicityParameterisation from_sigma_d(double sigma_d);

    static constexpr int num_parameters() {
        return 6;
    }

    const Vector6d &parameters() const;

    void set_parameters(const Vector6d &p);

    Matrix3d M() const;

    Matrix3d sigma() const;

    DerivativeMatrices first_derivatives() const;

    Mosaicity mosaicity() const;

    void print_mosaicity() const;

  private:
    Vector6d parameters_;
};