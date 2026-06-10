/**
 * @file predict.hpp
 * @brief Spot prediction algorithm (ReekeIndexGenerator-based) interface.
 */

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <cstdint>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/scan.hpp>
#include <gemmi/symmetry.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

inline constexpr size_t predicted_flag = (1 << 0);

struct predicted_data_rotation {
    // Shape {size, 3}
    std::vector<int> hkl;
    std::vector<double> s1;
    std::vector<double> xyz_px;
    std::vector<double> xyz_mm;
    // Shape {size, 1}
    std::vector<int> panels;
    std::vector<bool> enter;
    std::vector<size_t> flags;
    std::vector<int> ids;
    std::vector<uint64_t> experiment_ids;
    std::vector<std::string> identifiers;

    void add(const std::array<int, 3> &hkl_entry,
             const std::array<double, 3> &s1_entry,
             const std::array<double, 3> &xyz_px_entry,
             const std::array<double, 3> &xyz_mm_entry,
             int panel,
             bool enter_flag,
             size_t flag) {
        hkl.insert(hkl.end(), hkl_entry.begin(), hkl_entry.end());
        s1.insert(s1.end(), s1_entry.begin(), s1_entry.end());
        xyz_px.insert(xyz_px.end(), xyz_px_entry.begin(), xyz_px_entry.end());
        xyz_mm.insert(xyz_mm.end(), xyz_mm_entry.begin(), xyz_mm_entry.end());
        panels.push_back(panel);
        enter.push_back(enter_flag);
        flags.push_back(flag);
    }

    void merge(predicted_data_rotation &&other) {
        hkl.insert(hkl.end(), other.hkl.begin(), other.hkl.end());
        s1.insert(s1.end(), other.s1.begin(), other.s1.end());
        xyz_px.insert(xyz_px.end(), other.xyz_px.begin(), other.xyz_px.end());
        xyz_mm.insert(xyz_mm.end(), other.xyz_mm.begin(), other.xyz_mm.end());
        panels.insert(panels.end(), other.panels.begin(), other.panels.end());
        enter.insert(enter.end(), other.enter.begin(), other.enter.end());
        flags.insert(flags.end(), other.flags.begin(), other.flags.end());
    }
};

struct scan_varying_data {
    std::vector<Eigen::Vector3d> s0_at_scan_points;
    std::vector<Eigen::Matrix3d> A_at_scan_points;
    std::vector<Eigen::Matrix3d> r_setting_at_scan_points;

    auto operator<=>(const scan_varying_data &) const = default;
};

predicted_data_rotation predict_single_image(
  const int image_index,
  const scan_varying_data &sv_data,
  const Eigen::Vector3d s0,
  const Eigen::Matrix3d A,
  const Goniometer &goniometer,
  const Scan &scan,
  const Detector &detector,
  double param_dmin,
  gemmi::GroupOps crystal_symmetry_operations);

predicted_data_rotation predict_rotation(Experiment &experiment,
                                         const scan_varying_data &sv_data,
                                         double param_dmin,
                                         int buffer_size = 0,
                                         int nthreads = 1);

std::tuple<bool, scan_varying_data> extract_scan_varying_data(
  nlohmann::json elist_json_obj,
  Scan scan);
