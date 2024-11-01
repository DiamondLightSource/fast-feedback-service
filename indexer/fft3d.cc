#include <pocketfft_hdronly.h>
#include <map>
#include <stack>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <math.h>
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

#define _USE_MATH_DEFINES
#include <cmath>

using namespace pocketfft;

//std::tuple<std::vector<std::complex<double>>, std::vector<bool>>
void map_centroids_to_reciprocal_space_grid_cpp(
  std::vector<Vector3d> const& reciprocal_space_vectors,
  std::vector<std::complex<double>> &data_in,
  std::vector<bool> &selection,
  double d_min,
  double b_iso = 0) {
  const int n_points = 256;
  const double rlgrid = 2 / (d_min * n_points);
  const double one_over_rlgrid = 1 / rlgrid;
  const int half_n_points = n_points / 2;
  //std::vector<bool> selection(reciprocal_space_vectors.size(), true);

  //std::vector<std::complex<double>> data_in(256 * 256 * 256);
  for (int i = 0; i < reciprocal_space_vectors.size(); i++) {
    const Vector3d v = reciprocal_space_vectors[i];
    const double v_length = v.norm();
    const double d_spacing = 1 / v_length;
    if (d_spacing < d_min) {
      selection[i] = false;
      continue;
    }
    Vector3i coord;
    for (int j = 0; j < 3; j++) {
      coord[j] = ((int)round(v[j] * one_over_rlgrid)) + half_n_points;
    }
    if ((coord.maxCoeff() >= n_points) || coord.minCoeff() < 0) {
      selection[i] = false;
      continue;
    }
    double T;
    if (b_iso != 0) {
      T = std::exp(-b_iso * v_length * v_length / 4.0);
    } else {
      T = 1;
    }
    size_t index = coord[2] + (256 * coord[1]) + (256 * 256 * coord[0]);
    data_in[index] = {T, 0.0};
  }
  //return std::make_tuple(data_in, selection);
}

std::tuple<std::vector<double>, std::vector<bool>> fft3d(
  std::vector<Vector3d> const& reciprocal_space_vectors,
  double d_min,
  double b_iso = 0) {
  auto start = std::chrono::system_clock::now();

  std::vector<std::complex<double>> complex_data_in(256 * 256 * 256);
  std::vector<std::complex<double>> data_out(256 * 256 * 256);
  std::vector<double> real_out(256 * 256 * 256);
  std::vector<bool> used_in_indexing(reciprocal_space_vectors.size(), true);
  auto t1 = std::chrono::system_clock::now();

  ///std::vector<std::complex<double>> complex_data_in;
  ///std::vector<bool> used_in_indexing;
  ///std::tie(complex_data_in, used_in_indexing) =
  map_centroids_to_reciprocal_space_grid_cpp(reciprocal_space_vectors, complex_data_in, used_in_indexing, d_min, b_iso);
  auto t2 = std::chrono::system_clock::now();

  shape_t shape_in{256, 256, 256};
  stride_t stride_in{sizeof(std::complex<double>),
                     sizeof(std::complex<double>) * 256,
                     sizeof(std::complex<double>) * 256
                       * 256};  // must have the size of each element. Must have
                                // size() equal to shape_in.size()
  stride_t stride_out{sizeof(std::complex<double>),
                      sizeof(std::complex<double>) * 256,
                      sizeof(std::complex<double>) * 256
                        * 256};  // must have the size of each element. Must
                                 // have size() equal to shape_in.size()
  shape_t axes{0, 1, 2};         // 0 to shape.size()-1 inclusive
  bool forward{FORWARD};
  
  double fct{1.0f};
  size_t nthreads = 20;  // use all threads available - is this working?
  
  c2c(shape_in,
      stride_in,
      stride_out,
      axes,
      forward,
      complex_data_in.data(),
      data_out.data(),
      fct,
      nthreads);
  auto t3 = std::chrono::system_clock::now();

  for (int i = 0; i < real_out.size(); ++i) {
    real_out[i] = std::pow(data_out[i].real(), 2);
  }
  auto t4 = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = t4 - start;
  std::chrono::duration<double> elapsed_map = t2 - t1;
  std::chrono::duration<double> elapsed_make_arrays = t1 - start;
  std::chrono::duration<double> elapsed_c2c = t3 - t2;
  std::chrono::duration<double> elapsed_square = t4 - t3;
  std::cout << "Total time for fft3d: " << elapsed_seconds.count() << "s" << std::endl;

  std::cout << "elapsed time for making data arrays: " << elapsed_make_arrays.count() << "s" << std::endl;
  std::cout << "elapsed time for map_to_recip: " << elapsed_map.count() << "s" << std::endl;
  std::cout << "elapsed time for c2c: " << elapsed_c2c.count() << "s" << std::endl;
  std::cout << "elapsed time for squaring: " << elapsed_square.count() << "s" << std::endl;

  return std::make_tuple(real_out, used_in_indexing);
}