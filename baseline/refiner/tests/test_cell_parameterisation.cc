#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>
#include <dx2/crystal.hpp>
#include "cell_parameterisation.cc"
#include "gemmi/symmetry.hpp"
#include "gemmi/unitcell.hpp"

using Eigen::Vector3d;
using Eigen::Matrix3d;


TEST(BaselineIndexer, cell_parameterisation) {
    Vector3d a = {-0.19,2.78,6.05};
    Vector3d b = {0.09,-15.44,8.78};
    Vector3d c = {26.77,0.45,2.39};
    gemmi::SpaceGroup space_group = *gemmi::find_spacegroup_by_name("P1");
    Crystal crystal(a,b,c, space_group);
    CellParameterisation parameterisation(crystal);
    // Check the initial params are as expected.
    std::vector<double> p = parameterisation.get_params();
    std::vector<double> expected_p = {2278.037528258581, 319.6089400562122, 139.00920939203462, -71.8358313088007, -31.860812118145848, -6.218180236875039};
    for (int i=0;i<p.size();++i){
        EXPECT_NEAR(p[i], expected_p[i], 1E-12);
    }
    // Test updating the params
    std::vector<double> new_p = {2279,320,138,-71,-31,-6};
    parameterisation.set_params(new_p);
    p = parameterisation.get_params();
    for (int i=0;i<p.size();++i){
        EXPECT_NEAR(p[i], new_p[i], 1E-12);
    }
    // Now calculate the gradients and verify against DIALS values
    std::vector<Matrix3d> derivatives = parameterisation.get_dS_dp();
    Matrix3d expected_deriv_0({
        {3.329144216269376E-5, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(derivatives[0](i,j), expected_deriv_0(i,j), 1E-12);
        }
    }
    Matrix3d expected_deriv_1({
        {1.7044803613601804E-6, 0.0, 0.0},
        {2.000791372364517E-5, 8.84243975502925E-5, 0.0},
        {0.0, 0.0, 0.0}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(derivatives[1](i,j), expected_deriv_1(i,j), 1E-12);
        }
    }
    Matrix3d expected_deriv_2({
        {1.8303232869663895E-6, 0.0, 0.0},
        {1.7650788060680502E-6, 1.6715387060546735E-7, 0.0},
        {3.023521551413609E-5, 5.851977196284386E-6, 0.0001345954755145414}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(derivatives[2](i,j), expected_deriv_2(i,j), 1E-12);
        }
    }
    Matrix3d expected_deriv_3({
        {1.506580357865718E-5, 0.0, 0.0},
        {0.000176848795100585, 0.0, 0.0},
        {0.0, 0.0, 0.0}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(derivatives[3](i,j), expected_deriv_3(i,j), 1E-12);
        }
    }
    Matrix3d expected_deriv_4({
        {1.561205967796347E-5, 0.0, 0.0},
        {7.689078047851524E-6, 0.0, 0.0},
        {0.0002691909510290828, 0.0, 0.0}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(derivatives[4](i,j), expected_deriv_4(i,j), 1E-12);
        }
    }
    Matrix3d expected_deriv_5({
        {3.5325628643093795E-6, 0.0, 0.0},
        {4.146672183189763E-5, 7.689078047851526E-6, 0.0},
        {0.0, 0.0002691909510290828, 0.0}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(derivatives[5](i,j), expected_deriv_5(i,j), 1E-12);
        }
    }
    // Check the state
    Matrix3d state = parameterisation.get_state();
    Matrix3d expected_state({
        {0.15018874747345667, 0.0, 0.0},
        {-0.012794625871624953, 0.05654547996390009, 0.0},
        {-0.008344919481901558, -0.0016151457061744903, 0.03714835124201342}});
    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            EXPECT_NEAR(state(i,j), expected_state(i,j), 1E-12);
        }
    }
}