#include <gtest/gtest.h>
#include <math.h>

#include <Eigen/Dense>
#include "../ellipsoid_parameterisation.hpp"

using Eigen::Matrix3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

TEST(BaselineIntegrator, ellipsoid_parameterisation) {
    // Test that parameters update properly
    Vector6d p1 = {1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3};
    Simple6MosaicityParameterisation m_param(p1);
    Vector6d p2 = {2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3};
    m_param.set_parameters(p2);
    EXPECT_DOUBLE_EQ(m_param.parameters()[0], 2e-3);
    EXPECT_DOUBLE_EQ(m_param.parameters()[1], 3e-3);
    EXPECT_DOUBLE_EQ(m_param.parameters()[2], 4e-3);
    EXPECT_DOUBLE_EQ(m_param.parameters()[3], 5e-3);
    EXPECT_DOUBLE_EQ(m_param.parameters()[4], 6e-3);
    EXPECT_DOUBLE_EQ(m_param.parameters()[5], 7e-3);
    
    // check derivative calculation against finite differences
    Vector6d p;
    p << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;

    Simple6MosaicityParameterisation model(p);

    auto analytical = model.first_derivatives();

    constexpr double eps = 1e-7;

    for (int i = 0; i < 6; ++i) {

        Vector6d p_plus = p;
        Vector6d p_minus = p;

        p_plus(i) += eps;
        p_minus(i) -= eps;

        Eigen::Matrix3d numerical =
            (Simple6MosaicityParameterisation(p_plus).sigma() -
            Simple6MosaicityParameterisation(p_minus).sigma()) /
            (2.0 * eps);

        Eigen::Matrix3d diff = analytical[i] - numerical;

        EXPECT_LT(diff.cwiseAbs().maxCoeff(), 1e-8)
            << "Derivative " << i << " failed";
  }
}