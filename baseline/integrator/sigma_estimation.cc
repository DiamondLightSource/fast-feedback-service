#include <dx2/reflection.hpp>
#include <tuple>
#include <dx2/experiment.hpp>
#include <dx2/detector.hpp>
#include <dx2/beam.hpp>
#include "ffs_logger.hpp"

constexpr double RAD2DEG = 180.0 / M_PI;
constexpr size_t indexed_flag = (1 << 2);  // 4
/*
Function to calculate the square deviation in kabsch space between
the predicted and observed positions.
*/
double squaredev_in_kabsch_space(
  const Vector3d &xyzcal,//mm
  const Vector3d &xyzobs,//mm
  const Vector3d &s0,
  const Panel &panel) {
    Vector3d s1cal = panel.get_lab_coord(xyzcal[0], xyzcal[1]);
    Vector3d s1obs = panel.get_lab_coord(xyzobs[0], xyzobs[1]);

    Vector3d e1 = s1cal.cross(s0);
    e1.normalize();
    Vector3d e2 = s1cal.cross(e1);
    e2.normalize();
    double mags1 = std::sqrt(s1cal.dot(s1cal));
    Vector3d delta_s1 = s1obs - s1cal;
    double eps1 = e1.dot(delta_s1) / mags1;
    double eps2 = e2.dot(delta_s1) / mags1;
    double var = (eps1*eps1) + (eps2*eps2);
    return var;
}

std::tuple<double, double, double> estimate_sigmas(ReflectionTable const& indexed, Experiment<MonochromaticBeam>& expt, int min_bbox_depth = 6){
    auto flags = indexed.column<std::size_t>("flags");
    auto& flags_data = flags.value();
    std::vector<bool> selection(flags_data.extent(0), false);
    for (int i=0;i<flags_data.size();++i){
    if (flags_data(i,0) & indexed_flag){
        selection[i] = true;
    }
    }
    ReflectionTable filtered = indexed.select(selection);
    auto filtered_sigma_b = filtered.column<double>("sigma_b_variance");
    auto& filtered_sigma_b_data = filtered_sigma_b.value();
    auto filtered_sigma_m = filtered.column<double>("sigma_m_variance");
    auto& filtered_sigma_m_data = filtered_sigma_m.value();
    auto extent_z = filtered.column<int>("spot_extent_z");
    auto& extent_z_data = extent_z.value();
    double sigma_b_total = 0;
    double sigma_m_total = 0;
    int n_sigma_m = 0;
    for (int i=0;i<filtered_sigma_b_data.extent(0);++i){
        sigma_b_total += filtered_sigma_b_data(i,0);
        if (extent_z_data(i,0) >= min_bbox_depth){
            sigma_m_total += filtered_sigma_m_data(i,0);
            n_sigma_m++;
        }
    }
    double sigma_b_radians = std::pow(sigma_b_total / filtered_sigma_b_data.extent(0), 0.5);
    logger.info("Sigma b estimate (degrees): {:.6f} on {} reflections", RAD2DEG * sigma_b_radians, filtered_sigma_b_data.extent(0));
    if (n_sigma_m == 0){
        throw std::runtime_error("Unable to estimate sigma_m, no reflections above min_bbox_depth.");
    }
    double sigma_m_radians = std::pow(sigma_m_total / n_sigma_m, 0.5);
    logger.info("Sigma m estimate (degrees): {:.6f} on {} reflections with min_bbox_depth={}", sigma_m_radians * RAD2DEG, n_sigma_m, min_bbox_depth);
    // loop through refls - map s1 to recip space
    auto xyz = filtered.column<double>("xyzobs.mm.value");
    auto& xyzobs = xyz.value();
    auto xyz2 = filtered.column<double>("xyzcal.mm");
    auto& xyzcal = xyz2.value();
    Panel p = expt.detector().panels()[0];
    double tot_rmsd = 0;
    int count = 0;
    for (int i=0;i<xyzcal.extent(0);++i){
        Eigen::Map<Vector3d> xyzcal_this(&xyzcal(i,0));
        Eigen::Map<Vector3d> xyzobs_this(&xyzobs(i,0));
        double val = squaredev_in_kabsch_space(xyzcal_this, xyzobs_this, expt.beam().get_s0(), p);
        if (RAD2DEG * std::pow(val, 0.5) < 0.1){ // Guard against mispredictions in indexing.
            tot_rmsd += val;
            count++;
        }
    }
    if (count == 0){
        throw std::runtime_error("Unable to estimate rmsd deviation, predicted reflections are too far from observed");
    }
    double rmsd_deviation_radians = std::pow(tot_rmsd/count, 0.5);
    logger.info("Sigma rmsd (degrees): {:.6f} on {} reflections", rmsd_deviation_radians * RAD2DEG, count);
    return std::make_tuple(sigma_b_radians, sigma_m_radians, rmsd_deviation_radians);
}