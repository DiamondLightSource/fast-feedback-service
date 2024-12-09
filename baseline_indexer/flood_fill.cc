#include <map>
#include <stack>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <math.h>
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>

using Eigen::Vector3d;
using Eigen::Vector3i;

// Define a modulo function that returns python style modulo for negative numbers.
int modulo(int i, int n) {
  return (i % n + n) % n;
}

std::tuple<std::vector<int>,std::vector<Vector3d>>
flood_fill(std::vector<double> const& grid,
           double rmsd_cutoff = 15.0,
           int n_points = 256) {
  auto start = std::chrono::system_clock::now();
  //  First calc rmsd and use this to create a binary grid
  double sumg = 0.0;
  for (int i = 0; i < grid.size(); ++i) {
    sumg += grid[i];
  }
  double meang = sumg / grid.size();
  double sum_delta_sq = 0.0;
  for (int i = 0; i < grid.size(); ++i) {
    sum_delta_sq += std::pow(grid[i] - meang, 2);
  }
  double rmsd = std::pow(sum_delta_sq / grid.size(), 0.5);
  std::vector<int> grid_binary(n_points * n_points * n_points, 0);
  double cutoff = rmsd_cutoff * rmsd;
  for (int i = 0; i < grid.size(); i++) {
    if (grid[i] >= cutoff) {
      grid_binary[i] = 1;
    }
  }
  auto t2 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = t2 - start;
  std::cout << "Time for first part of flood fill: " << elapsed_time.count() << "s" << std::endl;

  // Now do flood fill. Wrap around the edge in all three dimensions.
  int n_voids = 0;
  std::stack<Vector3i> stack;
  std::vector<std::vector<Vector3i>> accumulators;
  int target = 1;
  int replacement = 2;
  std::vector<int> grid_points_per_void;
  int accumulator_index = 0;
  int total = n_points * n_points * n_points;
  int n_sq = n_points * n_points;
  int n_sq_minus_n = n_points * (n_points - 1);
  int nn_sq_minus_n = n_points * n_points * (n_points - 1);

  for (int i = 0; i < grid_binary.size(); i++) {
    if (grid_binary[i] == target) {
      // Convert the array index into xyz coordinates.
      // Store xyz coordinates on the stack, but index the array with 1D index.
      int x = i % n_points;
      int y = (i % n_sq) / n_points;
      int z = i / n_sq;
      Vector3i xyz = {x, y, z};
      stack.push(xyz);
      grid_binary[i] = replacement;
      std::vector<Vector3i> this_accumulator;
      accumulators.push_back(this_accumulator);
      n_voids++;
      grid_points_per_void.push_back(0);

      while (!stack.empty()) {
        Vector3i this_xyz = stack.top();
        stack.pop();
        accumulators[accumulator_index].push_back(this_xyz);
        grid_points_per_void[accumulator_index]++;

        int x_plus = this_xyz[0] + 1;
        int modx = modulo(this_xyz[0], n_points);
        int mody = modulo(this_xyz[1], n_points) * n_points;
        int modz = modulo(this_xyz[2], n_points) * n_sq;
        
        // For x,y,z, check locations +-1 on the grid and add to stack if match.

        int array_index = modulo(x_plus, n_points) + mody + modz;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          Vector3i new_xyz = {x_plus, this_xyz[1], this_xyz[2]};
          stack.push(new_xyz);
        }
        int x_minus = this_xyz[0] - 1;
        array_index = modulo(x_minus, n_points) + mody + modz;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          Vector3i new_xyz = {x_minus, this_xyz[1], this_xyz[2]};
          stack.push(new_xyz);
        }

        int y_plus = this_xyz[1] + 1;
        array_index = modx + (modulo(y_plus, n_points) * n_points) + modz;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          Vector3i new_xyz = {this_xyz[0], y_plus, this_xyz[2]};
          stack.push(new_xyz);
        }
        int y_minus = this_xyz[1] - 1;
        array_index = modx + (modulo(y_minus, n_points) * n_points) + modz;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          Vector3i new_xyz = {this_xyz[0], y_minus, this_xyz[2]};
          stack.push(new_xyz);
        }

        int z_plus = this_xyz[2] + 1;
        array_index = modx + mody + (modulo(z_plus, n_points) * n_sq);
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          Vector3i new_xyz = {this_xyz[0], this_xyz[1], z_plus};
          stack.push(new_xyz);
        }
        int z_minus = this_xyz[2] - 1;
        array_index = modx + mody + (modulo(z_minus, n_points) * n_sq);
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          Vector3i new_xyz = {this_xyz[0], this_xyz[1], z_minus};
          stack.push(new_xyz);
        }
      }
      replacement++;
      accumulator_index++;
    }
  }
  auto t3 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time2 = t3 - t2;
  std::cout << "Time for second part of flood fill: " << elapsed_time2.count() << "s" << std::endl;

  // Now calculate the unweighted centres of mass of each group.
  std::vector<Vector3d> centres_of_mass_frac(n_voids);
  for (int i = 0; i < accumulators.size(); i++) {
    std::vector<Vector3i> values = accumulators[i];
    int n = values.size();
    int divisor = n * n_points;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    for (int j = 0; j < n; j++) {
      x += values[j][0];
      y += values[j][1];
      z += values[j][2];
    }
    x /= divisor;
    y /= divisor;
    z /= divisor;
    centres_of_mass_frac[i] = {z, y, x};
  }
  return std::make_tuple(grid_points_per_void, centres_of_mass_frac);
}

std::tuple<std::vector<int>, std::vector<Vector3d>> flood_fill_filter(
  std::vector<int> grid_points_per_void,
  std::vector<Vector3d> centres_of_mass_frac,
   double peak_volume_cutoff = 0.15){
    // now filter out based on iqr range and peak_volume_cutoff
  std::vector<int> grid_points_per_void_unsorted(grid_points_per_void);
  std::sort(grid_points_per_void.begin(), grid_points_per_void.end());
  int Q3_index = grid_points_per_void.size() * 3 / 4;
  int Q1_index = grid_points_per_void.size() / 4;
  int iqr = grid_points_per_void[Q3_index] - grid_points_per_void[Q1_index];
  int iqr_multiplier = 5;
  int cut = (iqr * iqr_multiplier) + grid_points_per_void[Q3_index];

  while (grid_points_per_void[grid_points_per_void.size() - 1] > cut) {
    grid_points_per_void.pop_back();
  }
  int max_val = grid_points_per_void[grid_points_per_void.size() - 1];

  int peak_cutoff = (int)(peak_volume_cutoff * max_val);
  for (int i = grid_points_per_void_unsorted.size() - 1; i >= 0; i--) {
    if (grid_points_per_void_unsorted[i] <= peak_cutoff) {
      grid_points_per_void_unsorted.erase(grid_points_per_void_unsorted.begin() + i);
      centres_of_mass_frac.erase(centres_of_mass_frac.begin() + i);
    }
  }
  return std::make_tuple(grid_points_per_void_unsorted, centres_of_mass_frac);
}