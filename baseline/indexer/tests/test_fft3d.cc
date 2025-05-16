#include <gtest/gtest.h>
#include <math.h>

#include <Eigen/Dense>
#include <numeric>

#include "common.hpp"
#include "fft3d.cc"
using Eigen::Matrix3d;
using Eigen::Vector3d;

template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;

TEST(BaselineIndexer, map_centroids_to_reciprocal_space_test) {
    std::vector<double> reciprocal_space_vectors_data = {
      -0.2, 0.2, 0.25, -0.2, 0.1, 0.1};
    mdspan_type<double> reciprocal_space_vectors(
      reciprocal_space_vectors_data.data(),
      reciprocal_space_vectors_data.size() / 3,
      3);
    uint32_t n_points = 64;
    std::vector<std::complex<double>> complex_data_in(n_points * n_points * n_points);
    std::vector<bool> used_in_indexing(reciprocal_space_vectors.extent(0), true);
    double d_min = 2.0;
    // First test with no biso;
    double b_iso = 0.0;
    map_centroids_to_reciprocal_space_grid(reciprocal_space_vectors,
                                           complex_data_in,
                                           used_in_indexing,
                                           d_min,
                                           b_iso,
                                           n_points);
    // expect these map to data at indices 80294 and 80752
    EXPECT_DOUBLE_EQ(complex_data_in[80294].real(), 1.0);
    EXPECT_DOUBLE_EQ(complex_data_in[80752].real(), 1.0);
    // check no other values have been written
    EXPECT_DOUBLE_EQ(
      std::accumulate(
        complex_data_in.begin(), complex_data_in.end(), std::complex{0.0, 0.0})
        .real(),
      2.0);

    // Now set a biso;
    double b_iso_2 = 10.0;
    std::vector<std::complex<double>> complex_data_in_2(n_points * n_points * n_points);
    map_centroids_to_reciprocal_space_grid(reciprocal_space_vectors,
                                           complex_data_in_2,
                                           used_in_indexing,
                                           d_min,
                                           b_iso_2,
                                           n_points);
    EXPECT_DOUBLE_EQ(complex_data_in_2[80294].real(), 0.86070797642505781);
    EXPECT_DOUBLE_EQ(complex_data_in_2[80752].real(), 0.70029752396813894);
    //check no other values have been written
    EXPECT_DOUBLE_EQ(
      std::accumulate(
        complex_data_in_2.begin(), complex_data_in_2.end(), std::complex{0.0, 0.0})
        .real(),
      1.5610055003931969);

    // Now set a d_min, which filters out one of the points;
    double dmin_2 = 4.0;
    std::vector<std::complex<double>> complex_data_in_3(n_points * n_points * n_points);
    map_centroids_to_reciprocal_space_grid(reciprocal_space_vectors,
                                           complex_data_in_3,
                                           used_in_indexing,
                                           dmin_2,
                                           b_iso_2,
                                           n_points);
    // The index changes as reciprocal space is rescaled to cover the resolution range.
    // now expect a single value at index 27501
    EXPECT_DOUBLE_EQ(complex_data_in_3[27501].real(), 0.86070797642505781);
    // check no other values have been written
    EXPECT_DOUBLE_EQ(
      std::accumulate(
        complex_data_in_3.begin(), complex_data_in_3.end(), std::complex{0.0, 0.0})
        .real(),
      0.86070797642505781);
}