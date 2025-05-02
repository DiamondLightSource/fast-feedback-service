#include <dx2/crystal.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <gemmi/unitcell.hpp>
#include <iostream>
#include <optional>
#include <vector>

#include "combinations.cc"

using Eigen::Vector3d;

TEST(BaselineIndexer, combinations_test) {
    // With these vectors, we expect four combinations. Two fail the first test.
    std::vector<Vector3d> basis_vectors{
      {10.0, 0.0, 0.0}, {10.0, 1.0, 0.0}, {0.0, 2.5, 0.0}, {0.0, 0.0, 50.0}};
    // Set the max combinations to be greater than the actual number of potential combinations.
    CandidateOrientationMatrices candidates =
      CandidateOrientationMatrices(basis_vectors, 10);
    int count = 0;
    std::vector<gemmi::UnitCell> expected_cells;
    expected_cells.push_back({2.5, 10, 50, 90, 90, 90});
    expected_cells.push_back({2.5, 10.0499, 50, 90, 90, 95.7106});
    while (candidates.has_next()) {
        std::optional<Crystal> next_crystal = candidates.next();
        EXPECT_TRUE(
          next_crystal.has_value());  // Should always be true due to has_next check
        Crystal crystal = next_crystal.value();
        gemmi::UnitCell cell = crystal.get_unit_cell();
        EXPECT_NEAR(cell.a, expected_cells[count].a, 1e-4);
        EXPECT_NEAR(cell.b, expected_cells[count].b, 1e-4);
        EXPECT_NEAR(cell.c, expected_cells[count].c, 1e-4);
        EXPECT_NEAR(cell.alpha, expected_cells[count].alpha, 1e-4);
        EXPECT_NEAR(cell.beta, expected_cells[count].beta, 1e-4);
        EXPECT_NEAR(cell.gamma, expected_cells[count].gamma, 1e-4);
        count++;
    }
    // should be no more candidates
    std::optional<Crystal> next_crystal = candidates.next();
    EXPECT_FALSE(next_crystal.has_value());
}
