#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>
#include <dx2/crystal.hpp>
#include "orientation_parameterisation.cc"
//#include "gemmi/symmetry.hpp"
//#include "gemmi/unitcell.hpp"

using Eigen::Vector3d;
using Eigen::Matrix3d;


TEST(BaselineIndexer, beam_parameterisation) {
    Vector3d a = {-0.19,2.78,6.05};
    Vector3d b = {0.09,-15.44,8.78};
    Vector3d c = {26.77,0.45,2.39};
    gemmi::SpaceGroup space_group = *gemmi::find_spacegroup_by_name("P1");
    Crystal crystal(a,b,c, space_group);
    OrientationParameterisation parameterisation(crystal);
    // Check the initial params are as expected.
    std::vector<double> p = parameterisation.get_params();
    std::vector<double> expected_p = {0.0, 0.0, 0.0};
    for (int i=0;i<p.size();++i){
        EXPECT_DOUBLE_EQ(p[i], expected_p[i]);
    }
    // Test updating the params
    std::vector<double> new_p = {1.0,2.0, 20.0};
    parameterisation.set_params(new_p);
    p = parameterisation.get_params();
    for (int i=0;i<p.size();++i){
        EXPECT_DOUBLE_EQ(p[i], new_p[i]);
    }
    // Now calculate the gradients and verify against DIALS values
    std::vector<Matrix3d> derivatives = parameterisation.get_dS_dp();
    Matrix3d expected_deriv_0({
        {1.9005703928913567E-5, 6.51805884278694E-6, 4.931444834945913E-7},
        {-0.0009085102109894602, -0.00041676177137985016, -2.2778880009563593E-5},
        {0.0004164544536079352, -0.0009089610289886106, 1.8749289464301534E-5}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(derivatives[0](i,j), expected_deriv_0(i,j), 1E-12);
        }
    }
    Matrix3d expected_deriv_1({
        {0.0009085820893048154, 0.00041670948673358586, 2.078085636390427E-5},
        {1.8174065059392263E-5, 8.335301137794516E-6, 4.156725517629904E-7},
        {2.6707390077135652E-5, -8.38268165437287E-6, -0.0009996081462070772}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(derivatives[1](i,j), expected_deriv_1(i,j), 1E-12);
        }
    }
    Matrix3d expected_deriv_2({
        {-0.0004158378860448571, 0.000908613417947585, -3.873640736248012E-5},
        {-3.503059924492174E-5, 2.6559050185975988E-5, 0.000999033269701145},
        {0.0, 0.0, 0.0}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(derivatives[2](i,j), expected_deriv_2(i,j), 1E-12);
        }
    }
    // Check the state
    Matrix3d state = parameterisation.get_state();
    Matrix3d expected_state({
        {-0.03503059924492174, 0.02655905018597599, 0.999033269701145},
        {0.4158378860448571, -0.908613417947585, 0.03873640736248012},
        {0.9087638360136735, 0.41679284252350873, 0.020785013227984962}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(state(i,j), expected_state(i,j), 1E-12);
        }
    }
}