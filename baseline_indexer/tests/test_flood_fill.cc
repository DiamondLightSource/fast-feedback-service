#include <gtest/gtest.h>
#include <math.h>

#include <Eigen/Dense>
#include <numeric>

#include "common.hpp"
#include "flood_fill.cc"
using Eigen::Matrix3d;
using Eigen::Vector3d;

TEST(BaselineIndexer, flood_fill_test) {
    // Modify the test values (for channel) from cctbx masks flood_fill
    int n_points = 5;
    std::vector<double> grid(n_points * n_points * n_points, 0.0);
    std::vector<int> corner_cube{
      {0, 4, 20, 24, 100, 104, 120, 124}};  // cube across all 8 corners
    // i.e. at fractional coords (0,0,0), (0,0.8,0), (0,0,0.8), ... (0.8,0.8,0.8)
    std::vector<int> channel{
      {12, 37, 38, 39, 42, 43, 62, 63, 67, 112}};  // a channel with a break
    // channel: fractional coords along z: 1 at 0, 5 at 0.2, 3 at 0.4, 1 at 0.8 (==-0.2)
    for (auto& i : corner_cube) {
        grid[i] = 100;
    }
    for (auto& i : channel) {
        grid[i] = 100;
    }
    // now add a weak point (next to the corner), which should get filtered out by the rmsd cutoff filter.
    grid[1] = 1;  // the RMSD is approx 35, so anything below this is filtered out.
    std::vector<int> grid_points_per_void;
    std::vector<Vector3d> centres_of_mass_frac;
    std::tie(grid_points_per_void, centres_of_mass_frac) =
      flood_fill(grid, 1.0, n_points);

    EXPECT_EQ(grid_points_per_void[0], 10);  // channel
    EXPECT_EQ(grid_points_per_void[1], 8);   // corner cube
    // we return z,y,x.
    EXPECT_DOUBLE_EQ(centres_of_mass_frac[0][0], 1.2);   // z (== 0.2)
    EXPECT_DOUBLE_EQ(centres_of_mass_frac[0][1], 0.46);  // y
    EXPECT_DOUBLE_EQ(centres_of_mass_frac[0][2], 0.5);   // x
    // points over the boundary are equivalent modulo 1.0 (centre of space is at 0.5,0.5,0.5)
    EXPECT_DOUBLE_EQ(centres_of_mass_frac[1][0], 0.9);   // corner cube
    EXPECT_DOUBLE_EQ(centres_of_mass_frac[1][1], -0.1);  // corner cube (==0.9)
    EXPECT_DOUBLE_EQ(centres_of_mass_frac[1][2], 0.9);   // corner cube
}

TEST(BaselineIndexer, flood_fill_filter_test) {
    std::vector<int> grid_points_per_void{{1, 3, 1, 2, 80, 5, 3, 4, 2}};
    std::vector<Vector3d> centres_of_mass_frac;
    for (int i = 0; i < grid_points_per_void.size(); i++) {
        double frac = static_cast<double>(i + 1) / 10.0;
        centres_of_mass_frac.push_back({frac, frac, frac});
    }
    std::vector<int> grid_points_per_void_out;
    std::vector<Vector3d> centres_of_mass_frac_out;
    // The value 80 should be internally filtered out based on IQR
    // but solely for the purposes of the peak_volume_cutoff check.
    // i.e. Then max value is 5, so with a peak_volume_cutoff of 0.2,
    // only the ones should be removed.
    std::tie(grid_points_per_void_out, centres_of_mass_frac_out) =
      flood_fill_filter(grid_points_per_void, centres_of_mass_frac, 0.2);

    EXPECT_EQ(grid_points_per_void_out.size(), 7);
    EXPECT_EQ(centres_of_mass_frac_out.size(), 7);
    EXPECT_EQ(grid_points_per_void.size(), 9);  // it was unmodified.
    EXPECT_EQ(centres_of_mass_frac.size(), 9);  // it was unmodified.
    std::vector<int> expected_grid_points_per_void{{3, 2, 80, 5, 3, 4, 2}};
    std::vector<Vector3d> expected_centres_of_mass;
    expected_centres_of_mass.push_back({0.2, 0.2, 0.2});
    expected_centres_of_mass.push_back({0.4, 0.4, 0.4});
    expected_centres_of_mass.push_back({0.5, 0.5, 0.5});
    expected_centres_of_mass.push_back({0.6, 0.6, 0.6});
    expected_centres_of_mass.push_back({0.7, 0.7, 0.7});
    expected_centres_of_mass.push_back({0.8, 0.8, 0.8});
    expected_centres_of_mass.push_back({0.9, 0.9, 0.9});
    for (int i = 0; i < grid_points_per_void_out.size(); i++) {
        EXPECT_EQ(grid_points_per_void_out[i], expected_grid_points_per_void[i]);
        for (int j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(centres_of_mass_frac_out[i][j],
                             expected_centres_of_mass[i][j]);
        }
    }
}