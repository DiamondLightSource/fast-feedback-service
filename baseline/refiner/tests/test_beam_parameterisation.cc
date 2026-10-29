#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <dx2/beam.hpp>
#include <dx2/goniometer.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "beam_parameterisation.cc"

using Eigen::Vector3d;

TEST(BaselineIndexer, beam_parameterisation) {
    // Example models from c2sum test dataset
    MonochromaticBeam beam(1.23985);
    std::vector<Vector3d> axes;
    axes.push_back({1.0, 0.0, 0.0});
    axes.push_back({0.914, 0.279, -0.297});
    axes.push_back({1.0, 0.0, 0.0});
    std::vector<double> angles = {102.6, 37.9, 0.0};
    std::vector<std::string> names = {"PHI", "KAPPA", "OMEGA"};
    std::size_t scan_axis = 2;
    Goniometer goniometer(axes, angles, names, scan_axis);
    BeamParameterisation beam_param(beam, goniometer);
    // Check the initial params are as expected.
    std::vector<double> p = beam_param.get_params();
    std::vector<double> expected_p = {0.0, 0.0, 0.8065491793362101};
    for (int i = 0; i < p.size(); ++i) {
        EXPECT_DOUBLE_EQ(p[i], expected_p[i]);
    }
    // Test updating the params
    std::vector<double> new_p = {1.0, 2.0, 0.90};
    beam_param.set_params(new_p);
    p = beam_param.get_params();
    for (int i = 0; i < p.size(); ++i) {
        EXPECT_DOUBLE_EQ(p[i], new_p[i]);
    }
    // Now calculate the gradients and verify against DIALS values
    std::vector<Vector3d> derivatives = beam_param.get_dS_dp();
    Vector3d expected_deriv_0(
      {-1.7999985000004549E-9, 0.0008999995500000376, 8.999980500009076E-7});
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(derivatives[0][i], expected_deriv_0[i]);
    }
    Vector3d expected_deriv_1({0.0008999977500015377, 0.0, 1.7999979000009154E-6});
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(derivatives[1][i], expected_deriv_1[i]);
    }
    Vector3d expected_deriv_2(
      {0.001999997666667683, 0.0009999998333333415, -0.9999975000017084});
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(derivatives[2][i], expected_deriv_2[i]);
    }
    // Check the state
    Vector3d state = beam_param.get_state();
    Vector3d expected_state(
      {0.0017999979000009152, 0.0008999998500000073, -0.8999977500015376});
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(state[i], expected_state[i]);
    }
}