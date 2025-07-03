#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>
#include <dx2/beam.hpp>
#include <dx2/goniometer.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include "beam_parameterisation.cc"

using Eigen::Vector3d;


TEST(BaselineIndexer, beam_parameterisation) {
    std::string imported_expt = "/dls/i03/data/2024/cm37235-2/processing/JBE/c2sum/imported.expt";
    std::ifstream f(imported_expt);
    json elist_json_obj = json::parse(f);
    json beam_json = elist_json_obj["beam"][0];
    json goniometer_json = elist_json_obj["goniometer"][0];
    MonochromaticBeam beam(beam_json);
    Goniometer goniometer(goniometer_json);
    BeamParameterisation beam_param(beam, goniometer);
    // Check the initial params are as expected.
    std::vector<double> p = beam_param.get_params();
    std::vector<double> expected_p = {0.0, 0.0, 0.8065491793362101};
    for (int i=0;i<p.size();++i){
        EXPECT_DOUBLE_EQ(p[i], expected_p[i]);
    }
    // Test updating the params
    std::vector<double> new_p = {1.0,2.0, 0.90};
    beam_param.set_params(new_p);
    p = beam_param.get_params();
    for (int i=0;i<p.size();++i){
        EXPECT_DOUBLE_EQ(p[i], new_p[i]);
    }
    // Now calculate the gradients and verify against DIALS values
    std::vector<Vector3d> derivatives = beam_param.get_dS_dp();
    Vector3d expected_deriv_0({
        -1.7999985000004549E-9,
        0.0008999995500000376,
        8.999980500009076E-7});
    for (int i=0;i<3;++i){
        EXPECT_DOUBLE_EQ(derivatives[0][i], expected_deriv_0[i]);
    }
    Vector3d expected_deriv_1({
        0.0008999977500015377,
        0.0,
        1.7999979000009154E-6});
    for (int i=0;i<3;++i){
        EXPECT_DOUBLE_EQ(derivatives[1][i], expected_deriv_1[i]);
    }
    Vector3d expected_deriv_2({
        0.001999997666667683,
        0.0009999998333333415,
        -0.9999975000017084});
    for (int i=0;i<3;++i){
        EXPECT_DOUBLE_EQ(derivatives[2][i], expected_deriv_2[i]);
    }
    // Check the state
    Vector3d state = beam_param.get_state();
    Vector3d expected_state({
        0.0017999979000009152,
        0.0008999998500000073,
        -0.8999977500015376});
    for (int i=0;i<3;++i){
        EXPECT_DOUBLE_EQ(state[i], expected_state[i]);
    }
}