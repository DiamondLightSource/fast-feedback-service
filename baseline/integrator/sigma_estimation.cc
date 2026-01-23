#include <dx2/beam.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/reflection.hpp>
#include <math/math_utils.cuh>

#include "ffs_logger.hpp"

constexpr size_t indexed_flag = (1 << 2);  // 4
//constexpr size_t used_in_refinement_flag = (1 << 3);  // 8
/*
Function to calculate the square deviation in kabsch space between
the predicted and observed positions.
*/
std::pair<double, double> squaredev_in_kabsch_space(const Vector3d &xyzcal,  //mm
                                 const Vector3d &xyzobs,  //mm
                                 const Vector3d &s0,
                                 const Panel &panel,
                                 Vector3d m2) {
    Vector3d s1cal = panel.get_lab_coord(xyzcal[0], xyzcal[1]);
    Vector3d s1obs = panel.get_lab_coord(xyzobs[0], xyzobs[1]);
    double delta_phi = xyzcal[2] - xyzobs[2];
    Vector3d e1 = s1cal.cross(s0);
    e1.normalize();
    Vector3d e2 = s1cal.cross(e1);
    e2.normalize();
    double zeta = m2.dot(e1);
    double mags1 = std::sqrt(s1cal.dot(s1cal));
    Vector3d delta_s1 = s1obs - s1cal;
    double eps1 = e1.dot(delta_s1) / mags1;
    double eps2 = e2.dot(delta_s1) / mags1;
    double eps3 = delta_phi * zeta;
    double varxy = (eps1 * eps1) + (eps2 * eps2);
    double varz = eps3 * eps3;
    return std::make_pair(varxy, varz);
}

std::pair<double, double> estimate_sigmas(ReflectionTable const &indexed,
                                                   Experiment<MonochromaticBeam> &expt,
                                                   int min_bbox_depth = 6) {
    auto flags = indexed.column<std::size_t>("flags");
    auto &flags_data = flags.value();
    std::vector<bool> selection(flags_data.extent(0), false);
    for (int i = 0; i < flags_data.size(); ++i) {
        // FIXME - once refinement fully implemented, use used_in_refinement_flag instead.
        if (flags_data(i, 0) & indexed_flag) {
            selection[i] = true;
        }
    }
    ReflectionTable filtered = indexed.select(selection);
    auto filtered_sigma_b = filtered.column<double>("sigma_b_variance");
    auto &filtered_sigma_b_data = filtered_sigma_b.value();
    auto filtered_sigma_m = filtered.column<double>("sigma_m_variance");
    auto &filtered_sigma_m_data = filtered_sigma_m.value();
    auto extent_z = filtered.column<int>("spot_extent_z");
    auto &extent_z_data = extent_z.value();
    double sigma_b_total = 0;
    double sigma_m_total = 0;
    int n_sigma_m = 0;
    for (int i = 0; i < filtered_sigma_b_data.extent(0); ++i) {
        sigma_b_total += filtered_sigma_b_data(i, 0);
        if (extent_z_data(i, 0) >= min_bbox_depth) {
            sigma_m_total += filtered_sigma_m_data(i, 0);
            n_sigma_m++;
        }
    }
    double sigma_b_radians =
      std::pow(sigma_b_total / filtered_sigma_b_data.extent(0), 0.5);
    logger.info("Internal sigma-b estimate (degrees): {:.6f} on {} reflections",
                radians_to_degrees(sigma_b_radians),
                filtered_sigma_b_data.extent(0));
    if (n_sigma_m == 0) {
        throw std::runtime_error(
          "Unable to estimate sigma_m, no reflections above min_bbox_depth.");
    }
    double sigma_m_radians = std::pow(sigma_m_total / n_sigma_m, 0.5);
    logger.info(
      "Internal sigma-m estimate (degrees): {:.6f} on {} reflections with min_bbox_depth={}",
      radians_to_degrees(sigma_m_radians),
      n_sigma_m,
      min_bbox_depth);
    // loop through refls - map s1 to recip space
    auto xyz = filtered.column<double>("xyzobs.mm.value");
    auto &xyzobs = xyz.value();
    auto xyz2 = filtered.column<double>("xyzcal.mm");
    auto &xyzcal = xyz2.value();
    auto s1_ = filtered.column<double>("s1");
    auto &s1 = s1_.value();
    Panel p = expt.detector().panels()[0];
    double tot_rmsd = 0;
    int count = 0;
    constexpr double deg_to_rad = M_PI / 180.0;
    double tot_rmsd_z = 0;
    Vector3d s0 = expt.beam().get_s0();
    Vector3d m2 = expt.goniometer().get_rotation_axis();
    for (int i = 0; i < xyzcal.extent(0); ++i) {
        Eigen::Map<Vector3d> xyzcal_this(&xyzcal(i, 0));
        Eigen::Map<Vector3d> xyzobs_this(&xyzobs(i, 0));
        auto [valxy, valz] = squaredev_in_kabsch_space(xyzcal_this, xyzobs_this, s0, p, m2);
        if (radians_to_degrees(std::pow(valxy, 0.5))
            < 0.1) {  // Guard against mispredictions in indexing.
            tot_rmsd += valxy;
            tot_rmsd_z += valz;
            count++;
        }
    }
    if (count == 0) {
        throw std::runtime_error(
          "Unable to estimate rmsd deviation, predicted reflections are too far from "
          "observed");
    }
    double rmsd_deviation_radians = std::pow(tot_rmsd / count, 0.5);
    logger.info("Misprediction sigma-b estimate (degrees): {:.6f} on {} reflections",
                radians_to_degrees(rmsd_deviation_radians),
                count);
    double rmsdz_deviation_radians = std::pow(tot_rmsd_z / count, 0.5);
    logger.info("Misprediction sigma-m estimate (degrees): {:.6f} on {} reflections",
                radians_to_degrees(rmsdz_deviation_radians),
                count);
    // Total sigma given by sqrt of sum of variances
    double total_sigma_b = std::pow(std::pow(sigma_b_radians,2) + std::pow(rmsd_deviation_radians, 2), 0.5);
    double total_sigma_m = std::pow(std::pow(sigma_m_radians,2) + std::pow(rmsdz_deviation_radians, 2), 0.5);
    logger.info("Overall sigma-b estimate (degrees): {:.6f}",
                radians_to_degrees(total_sigma_b));
    logger.info("Overall sigma-m estimate (degrees): {:.6f}",
                radians_to_degrees(total_sigma_m));
    return std::make_pair(total_sigma_b, total_sigma_m);
}