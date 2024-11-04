#include <map>
#include <stack>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <math.h>
#include <iostream>

using Eigen::Vector3d;


#define _USE_MATH_DEFINES
#include <cmath>

class VectorGroup {
public:
  void add(Vector3d vec, int weight) {
    vectors.push_back(vec);
    weights.push_back(weight);
  }
  Vector3d mean() {
    int n = vectors.size();
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    for (const Vector3d& i : vectors) {
      sum_x += i[0];
      sum_y += i[1];
      sum_z += i[2];
    }
    Vector3d m = {sum_x / n, sum_y / n, sum_z / n};
    return m;
  }
  std::vector<Vector3d> vectors{};
  std::vector<int> weights{};
};

struct SiteData {
  Vector3d site;
  double length;
  int volume;
};
bool compare_site_data(const SiteData& a, const SiteData& b) {
  return a.length < b.length;
}
bool compare_site_data_volume(const SiteData& a, const SiteData& b) {
  return a.volume > b.volume;
}

double vector_length(Vector3d v) {
  return std::pow(std::pow(v[0], 2) + std::pow(v[1], 2) + std::pow(v[2], 2), 0.5);
}

double angle_between_vectors_degrees(Vector3d v1, Vector3d v2) {
  double l1 = vector_length(v1);
  double l2 = vector_length(v2);
  double dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  double normdot = dot / (l1 * l2);
  if (std::abs(normdot - 1.0) < 1E-6) {
    return 0.0;
  }
  if (std::abs(normdot + 1.0) < 1E-6) {
    return 180.0;
  }
  double angle = std::acos(normdot) * 180.0 / M_PI;
  return angle;
}

bool is_approximate_integer_multiple(Vector3d v1,
                                     Vector3d v2,
                                     double relative_length_tolerance = 0.2,
                                     double angular_tolerance = 5.0) {
  double angle = angle_between_vectors_degrees(v1, v2);
  if ((angle < angular_tolerance) || (std::abs(180 - angle) < angular_tolerance)) {
    double l1 = vector_length(v1);
    double l2 = vector_length(v2);
    if (l1 > l2) {
      std::swap(l1, l2);
    }
    double n = l2 / l1;
    if (std::abs(std::round(n) - n) < relative_length_tolerance) {
      return true;
    }
  }
  return false;
}

std::vector<Vector3d> sites_to_vecs(
  std::vector<Vector3d> centres_of_mass_frac,
  std::vector<int> grid_points_per_void,
  double d_min,
  double min_cell = 3.0,
  double max_cell = 92.3) {
  auto start = std::chrono::system_clock::now();

  int n_points = 256;
  double fft_cell_length = n_points * d_min / 2.0;
  // sites_mod_short and convert to cartesian
  for (int i = 0; i < centres_of_mass_frac.size(); i++) {
    for (size_t j = 0; j < 3; j++) {
      if (centres_of_mass_frac[i][j] > 0.5) {
        centres_of_mass_frac[i][j]--;
      }
      centres_of_mass_frac[i][j] *= fft_cell_length;
    }
  }

  // now do some filtering
  std::vector<SiteData> filtered_data;
  for (int i = 0; i < centres_of_mass_frac.size(); i++) {
    auto v = centres_of_mass_frac[i];
    double length =
      std::pow(std::pow(v[0], 2) + std::pow(v[1], 2) + std::pow(v[2], 2), 0.5);
    if ((length > min_cell) && (length < 2 * max_cell)) {
      SiteData site_data = {centres_of_mass_frac[i], length, grid_points_per_void[i]};
      filtered_data.push_back(site_data);
    }
  }
  // now sort filtered data

  // need to sort volumes and sites by length for group_vectors, and also filter by max
  // and min cell
  //std::stable_sort(filtered_data.begin(), filtered_data.end(), compare_site_data);

  // now 'group vectors'
  double relative_length_tolerance = 0.1;
  double angular_tolerance = 5.0;
  std::vector<VectorGroup> vector_groups{};
  for (int i = 0; i < filtered_data.size(); i++) {
    bool matched_group = false;
    double length = filtered_data[i].length;
    for (int j = 0; j < vector_groups.size(); j++) {
      Vector3d mean_v = vector_groups[j].mean();
      double mean_v_length = vector_length(mean_v);
      if ((std::abs(mean_v_length - length) / std::max(mean_v_length, length))
          < relative_length_tolerance) {
        double angle = angle_between_vectors_degrees(mean_v, filtered_data[i].site);
        if (angle < angular_tolerance) {
          vector_groups[j].add(filtered_data[i].site, filtered_data[i].volume);
          matched_group = true;
          break;
        } else if (std::abs(180 - angle) < angular_tolerance) {
          vector_groups[j].add(-1.0 * filtered_data[i].site, filtered_data[i].volume);
          matched_group = true;
          break;
        }
      }
    }
    if (!matched_group) {
      VectorGroup group = VectorGroup();
      group.add(filtered_data[i].site, filtered_data[i].volume);
      vector_groups.push_back(group);
    }
  }
  std::vector<SiteData> grouped_data;
  for (int i = 0; i < vector_groups.size(); i++) {
    Vector3d site = vector_groups[i].mean();
    int max = *std::max_element(vector_groups[i].weights.begin(),
                                vector_groups[i].weights.end());
    SiteData site_data = {site, vector_length(site), max};
    grouped_data.push_back(site_data);
  }
  std::stable_sort(grouped_data.begin(), grouped_data.end(), compare_site_data_volume);
  std::stable_sort(grouped_data.begin(), grouped_data.end(), compare_site_data);

  // std::vector<Vector3d> unique_vectors;
  // std::vector<int> unique_volumes;
  std::vector<SiteData> unique_sites;
  for (int i = 0; i < grouped_data.size(); i++) {
    bool is_unique = true;
    Vector3d v = grouped_data[i].site;
    for (int j = 0; j < unique_sites.size(); j++) {
      if (unique_sites[j].volume > grouped_data[i].volume) {
        if (is_approximate_integer_multiple(unique_sites[j].site, v)) {
          std::cout << "rejecting " << vector_length(v) << ": is integer multiple of "
                    << vector_length(unique_sites[j].site) << std::endl;
          is_unique = false;
          break;
        }
      }
    }
    if (is_unique) {
      // std::cout << v[0] << " " << v[1] << " " << v[2] << std::endl;
      // unique_vectors.push_back(v);
      // unique_volumes.push_back(grouped_data[i].volume);
      SiteData site{v, vector_length(v), grouped_data[i].volume};
      unique_sites.push_back(site);
    }
  }
  // now sort by peak volume again
  std::stable_sort(unique_sites.begin(), unique_sites.end(), compare_site_data_volume);
  std::vector<Vector3d> unique_vectors_sorted;
  std::cout << "Candidate basis vectors: " << std::endl;
  for (int i = 0; i < unique_sites.size(); i++) {
    unique_vectors_sorted.push_back(unique_sites[i].site);
    std::cout << i << " " << unique_sites[i].length << " " << unique_sites[i].volume << std::endl;
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time for sites_to_vecs: " << elapsed_seconds.count() << "s"
            << std::endl;
  return unique_vectors_sorted;
}