#include <gtest/gtest.h>
#include <math.h>

#include <Eigen/Dense>
#include <numeric>

#include "common.hpp"
#include "peaks_to_rlvs.cc"

using Eigen::Matrix3d;
using Eigen::Vector3d;

TEST(BaselineIndexer, peaks_to_rlvs_test) {
    // peaks_to_rlvs transforms the fractional centres of mass on the FFT grid
    // back into basis vectors in reciprocal space.
    // The origin of space is at fractional coordinate (0.5, 0.5, 0.5).
    std::vector<int> grid_points_per_void;
    std::vector<Vector3d> centres_of_mass_frac;
    grid_points_per_void.push_back(8);
    grid_points_per_void.push_back(10);
    grid_points_per_void.push_back(10);
    centres_of_mass_frac.push_back(
      {0.75, 0.75, 0.75});  // rlp = (64.0, 64.0, 64.0) = 110.85A
    centres_of_mass_frac.push_back(
      {0.1, 0.1, 0.1});  // rlp = (25.6, 25.6,25.6) = 44.34A
    centres_of_mass_frac.push_back(
      {0.4, 0.4, 0.4});  // rlp = (102.4, 102.4,102.4) = 177.36A
    double dmin = 2.0;
    double min_cell = 3.0;
    double max_cell = 100.0;
    uint32_t n_points = 256;
    std::vector<Vector3d> unique = peaks_to_rlvs(
      centres_of_mass_frac, grid_points_per_void, dmin, min_cell, max_cell, n_points);
    // Results should be sorted by grid points per void in descending order
    // Equivalent multiples are not filtered if the grid points per void are equal.
    EXPECT_EQ(unique.size(), 3);
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(unique[0][i], 25.6);  // point at 0.1,0.1,0.1
    }
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(unique[1][i], 102.4);  // point at 0.4,0.4,0.4
    }
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(unique[2][i], -64.0);  // point at 0.75,0.75,0.75
    }

    grid_points_per_void[1] = 11;  // This causes the third to
    // get filtered as an equivalent to the second
    std::vector<Vector3d> unique2 = peaks_to_rlvs(
      centres_of_mass_frac, grid_points_per_void, dmin, min_cell, max_cell, n_points);
    EXPECT_EQ(unique2.size(), 2);
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(unique2[0][i], 25.6);  // point at 0.1,0.1,0.1
    }
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(unique2[1][i], -64.0);  // point at 0.75,0.75,0.75
    }

    // now check grouping based on length and angle - second and third should be combined
    centres_of_mass_frac[1] = {0.6, 0.6, 0.6};  // i.e. inverse pair to index 2
    centres_of_mass_frac[2] = {
      0.405, 0.405, 0.405};  // similar but not exactly equal to index 1
    grid_points_per_void[1] =
      10;  // i.e. wouldn't get filtered out by approx multiple filter
    std::vector<Vector3d> unique3 = peaks_to_rlvs(
      centres_of_mass_frac, grid_points_per_void, dmin, min_cell, max_cell, n_points);
    EXPECT_EQ(unique3.size(), 2);
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(
          unique3[0][i],
          -103.04);  // mean of point at 0.6,0.6,0.6 and 0.405,0.405,0.405
    }
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(unique3[1][i], -64.0);  // point at 0.75,0.75,0.75
    }

    // check min and max cell filters
    // the three points are at d-values of 44.34, 177.36 and 110.85
    // set min_cell to 50 and max_cell to 80, to just leave the 110.85 solution (yes the filter in the
    // dials code is cell < 2 * max_cell...). This could be changed here to make sense.
    double min_cell2 = 50.0;
    double max_cell2 = 80.0;
    centres_of_mass_frac[2] = {0.4, 0.4, 0.4};
    std::vector<Vector3d> unique4 = peaks_to_rlvs(
      centres_of_mass_frac, grid_points_per_void, dmin, min_cell2, max_cell2, n_points);
    EXPECT_EQ(unique4.size(), 1);
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(unique4[0][i], -64.0);  // rlp = (64.0, 64.0, 64.0) = 110.85A
    }
}