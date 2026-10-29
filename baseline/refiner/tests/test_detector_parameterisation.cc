#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <dx2/detector.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "detector_parameterisation.cc"

using Eigen::Matrix3d;
using json = nlohmann::json;

// Example models from c2sum test dataset
json panel_data = {{"name", "Panel"},
                   {"type", "SENSOR_PAD"},
                   {"fast_axis", {1.0, 0.0, 0.0}},
                   {"slow_axis", {0.0, -1.0, 0.0}},
                   {"origin", {-218.66704000000001, 209.22080000000003, -170.0}},
                   {"raw_image_offset", {0, 0}},
                   {"image_size", {2463, 2527}},
                   {"pixel_size", {0.17200000000000001, 0.17200000000000001}},
                   {"trusted_range", {0.0, 1224245.0}},
                   {"thickness", 0.32},
                   {"material", "Si"},
                   {"mu", 7.896517868233737},
                   {"identifier", ""},
                   {"gain", 1.0},
                   {"pedestal", 0.0},
                   {"px_mm_strategy", {{"type", "ParallaxCorrectedPxMmStrategy"}}}};

TEST(BaselineIndexer, detector_parameterisation) {
    Panel panel(panel_data);
    DetectorParameterisation detector_param(panel);
    // Check the initial params are as expected.
    std::vector<double> p = detector_param.get_params();
    std::vector<double> expected_p = {170.0, -6.84904, 8.1012, 0.0, 0.0, 0.0};
    for (int i = 0; i < p.size(); ++i) {
        EXPECT_DOUBLE_EQ(p[i], expected_p[i]);
    }
    // Test updating the params
    std::vector<double> new_p = {169.0, -6.8, 7.0, 60, 50, 30};
    detector_param.set_params(new_p);
    p = detector_param.get_params();
    for (int i = 0; i < p.size(); ++i) {
        EXPECT_DOUBLE_EQ(p[i], new_p[i]);
    }
    // Now calculate the gradients and verify against DIALS values
    std::vector<Matrix3d> derivatives = detector_param.get_dS_dp();
    Matrix3d expected_deriv_0({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(derivatives[0](i, j), expected_deriv_0(i, j));
        }
    }
    Matrix3d expected_deriv_1({{0.0, 0.0, 0.9978412784317173},
                               {0.0, 0.0, -0.05988906708567075},
                               {0.0, 0.0, 0.026945921794682914}});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(derivatives[1](i, j), expected_deriv_1(i, j));
        }
    }
    Matrix3d expected_deriv_2({{0.0, 0.0, -0.058440572179157796},
                               {0.0, 0.0, -0.9969530491866815},
                               {0.0, 0.0, -0.05166543564852367}});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(derivatives[2](i, j), expected_deriv_2(i, j));
        }
    }
    Matrix3d expected_deriv_3(
      {{-5.844057217915779E-5, -0.0009978412784317173, 0.22264413437097877},
       {-0.0009969530491866813, 5.988906708567075E-5, 0.20535589333950147},
       {-5.1665435648523665E-5, -2.6945921794682916E-5, 0.016962314374310245}});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(derivatives[3](i, j), expected_deriv_3(i, j));
        }
    }
    Matrix3d expected_deriv_4(
      {{1.7964025238955129E-6, 2.9904105388757763E-5, -0.006682217180543301},
       {2.996951229984214E-6, 4.988923375150408E-5, -0.011147990905080535},
       {-5.986211902667757E-5, -0.0009965044539607033, 0.22267374650329727}});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(derivatives[4](i, j), expected_deriv_4(i, j));
        }
    }
    Matrix3d expected_deriv_5(
      {{-2.6945921794682923E-5, 5.1665435648523665E-5, -0.004975514225558792},
       {0.0, 0.0, 0.0},
       {0.0009978412784317173, -5.8440572179157796E-5, -0.20585472658632034}});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(derivatives[5](i, j), expected_deriv_5(i, j));
        }
    }
    // Check the state
    Matrix3d state = detector_param.get_state();
    Matrix3d expected_state(
      {{0.9978412784317169, -0.05844057217915775, -205.85472658632028},
       {-0.059889067085670725, -0.9969530491866812, 222.77398727917634},
       {0.026945921794682904, -0.05166543564852365, -164.02448577444122}});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(state(i, j), expected_state(i, j));
        }
    }
}