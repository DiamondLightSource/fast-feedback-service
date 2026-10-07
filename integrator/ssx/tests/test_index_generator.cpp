#include <gtest/gtest.h>
#include <math.h>

#include <Eigen/Dense>
#include <gemmi/symmetry.hpp>
#include <gemmi/unitcell.hpp>
#include <chrono>

#include "predictor/index_generators.hpp"

using Eigen::Vector3i;

TEST(SSXIntegrator, ssx_index_generator) {

    gemmi::SpaceGroup space_group = *gemmi::find_spacegroup_by_name("P 21 3");
    gemmi::GroupOps crystal_symmetry_operations = space_group.operations();
    gemmi::UnitCell cell = {96.41, 96.41, 96.41, 90.0, 90.0, 90.0};

    auto t1 = std::chrono::system_clock::now();
    IndexGenerator g(cell, crystal_symmetry_operations, 1.58);
    std::vector<Vector3i> indices = g.to_array();
    auto t2 = std::chrono::system_clock::now();

    std::cout << indices.size() << std::endl;
    std::chrono::duration<double> elapsed_time = t2 - t1;
    std::cout << "Total time for index generator: " << elapsed_time.count() << std::endl;
    ASSERT_EQ(indices.size(), 951424);
}