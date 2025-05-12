#include <dx2/beam.h>
#include <dx2/crystal.h>
#include <dx2/detector.h>
#include <dx2/goniometer.h>
#include <dx2/scan.h>
#include <dx2/reflection.hpp>

#include <Eigen/Dense>
#include <iostream>
#include <random>

#include "scan_static_predictor.cc"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

constexpr double iqr_multiplier = 3.0;
// imported from scanstaticpredictor constexpr size_t predicted_value = (1 << 0); //predicted flag
constexpr size_t used_in_refinement_value = (1 << 3);  //used in refinement flag
constexpr size_t centroid_outlier_value = (1 << 17);   //centroid outlier flag
constexpr size_t overloaded_value = (1 << 10);         //overloaded flag

/**
 * @brief Generate a random selection of indices from a population up to a given sample size.
 * @param pop_size The size of the population that is to be selected from.
 * @param sample_size The number of random elements to select.
 * @param seed The seed value for the mt19937 Mersenne-Twister pseudo-random generator.
 * @returns A vector of indices.
*/
std::vector<std::size_t> random_selection(int pop_size,
                                          int sample_size,
                                          int seed = 43) {
    std::mt19937 mt(seed);
    std::vector<size_t> result(pop_size);
    std::vector<size_t>::iterator r = result.begin();
    for (size_t i = 0; i < pop_size; i++) {
        *r++ = i;
    }
    r = result.begin();
    for (std::size_t i = 0; i < pop_size; i++) {
        std::size_t j = static_cast<std::size_t>(mt()) % pop_size;
        std::swap(r[i], r[j]);
    }
    result.resize(sample_size);
    std::sort(result.begin(), result.end());
    return result;
}

/**
 * @brief Perform Tukey outlier rejection on 3D residuals, using an iqr multiplier of 3.
 * @param xresid A vector of residuals in the first dimension.
 * @param yresid A vector of residuals in the second dimension.
 * @param zresid A vector of residuals in the third dimension.
 * @returns A vector of outlier indices.
 */
std::vector<size_t> simple_tukey(std::vector<double> xresid,
                                 std::vector<double> yresid,
                                 std::vector<double> zresid) {
    std::vector<size_t> sel{};
    std::vector<double> xresid_unsorted(xresid.begin(), xresid.end());
    std::vector<double> yresid_unsorted(yresid.begin(), yresid.end());
    std::vector<double> zresid_unsorted(zresid.begin(), zresid.end());
    std::sort(xresid.begin(), xresid.end());
    std::sort(yresid.begin(), yresid.end());
    std::sort(zresid.begin(), zresid.end());

    // this is the way scitbx.math.five_number_summary does iqr (which matches R)
    int n_lower = 0;
    double Q1x, Q1y, Q1z, Q3x, Q3y, Q3z;
    int n = xresid.size();
    int upper_start = n / 2;
    if (n % 2) {
        n_lower = (n / 2) + 1;
    } else {
        n_lower = n / 2;
    }
    if (n_lower % 2) {
        Q1x = xresid[n_lower / 2];
        Q3x = xresid[upper_start + (n_lower / 2)];
        Q1y = yresid[n_lower / 2];
        Q3y = yresid[upper_start + (n_lower / 2)];
        Q1z = zresid[n_lower / 2];
        Q3z = zresid[upper_start + (n_lower / 2)];
    } else {
        Q1x = (xresid[n_lower / 2] + xresid[(n_lower / 2) - 1]) / 2;
        Q3x = (xresid[upper_start + (n_lower / 2)]
               + xresid[upper_start + (n_lower / 2) - 1])
              / 2;
        Q1y = (yresid[n_lower / 2] + yresid[(n_lower / 2) - 1]) / 2;
        Q3y = (yresid[upper_start + (n_lower / 2)]
               + yresid[upper_start + (n_lower / 2) - 1])
              / 2;
        Q1z = (zresid[n_lower / 2] + zresid[(n_lower / 2) - 1]) / 2;
        Q3z = (zresid[upper_start + (n_lower / 2)]
               + zresid[upper_start + (n_lower / 2) - 1])
              / 2;
    }
    double iqrx = Q3x - Q1x;
    double uppercutx = (iqrx * iqr_multiplier) + Q3x;
    double lowercutx = Q1x - (iqrx * iqr_multiplier);

    double iqry = Q3y - Q1y;
    double uppercuty = (iqry * iqr_multiplier) + Q3y;
    double lowercuty = Q1y - (iqry * iqr_multiplier);

    double iqrz = Q3z - Q1z;
    double uppercutz = (iqrz * iqr_multiplier) + Q3z;
    double lowercutz = Q1z - (iqrz * iqr_multiplier);

    auto is_outlier = [](double value, double upper, double lower) -> bool {
        return value > upper || value < lower;
    };  // helper lambda function

    for (int i = 0; i < xresid_unsorted.size(); i++) {
        if (is_outlier(xresid_unsorted[i], uppercutx, lowercutx)
            || is_outlier(yresid_unsorted[i], uppercuty, lowercuty)
            || is_outlier(zresid_unsorted[i], uppercutz, lowercutz)) {
            sel.push_back(i);
        }
    }
    return sel;
}

/**
 * @brief Perform outlier rejection on a reflection table based on centroid residuals.
 * @param ReflectionTable The reflection table.
 * @returns A reflection table that is a subset of the input table.
 */
ReflectionTable outlier_filter(ReflectionTable& reflections) {
    // First make sure the reflections have the predicted flag.
    auto flags_ = reflections.column<std::size_t>("flags");
    auto& flags = flags_.value();
    auto xyzobs_ = reflections.column<double>("xyzobs_mm");
    const auto& xyzobs_mm = xyzobs_.value();
    auto xyzcal_ = reflections.column<double>("xyzcal_mm");
    const auto& xyzcal_mm = xyzcal_.value();

    std::vector<double> x_resid(flags.size(), 0.0);
    std::vector<double> y_resid(flags.size(), 0.0);
    std::vector<double> phi_resid(flags.size(), 0.0);
    std::vector<std::size_t> sel{};

    for (int i = 0; i < flags.size(); i++) {
        if ((flags(i,0) & predicted_value) == predicted_value) {
            x_resid[i] = xyzcal_mm(i, 0) - xyzobs_mm(i, 0);
            y_resid[i] = xyzcal_mm(i, 1) - xyzobs_mm(i, 1);
            phi_resid[i] = xyzcal_mm(i, 2) - xyzobs_mm(i, 2);
            sel.push_back(i);
        }
    }
    // Avoid selecting on the whole table, just the data of interest.
    std::vector<double> x_resid_sel(sel.size());
    std::vector<double> y_resid_sel(sel.size());
    std::vector<double> phi_resid_sel(sel.size());
    for (int i = 0; i < sel.size(); i++) {
        int idx = sel[i];
        x_resid_sel[i] = x_resid[idx];
        y_resid_sel[i] = y_resid[idx];
        phi_resid_sel[i] = phi_resid[idx];
    }

    // Do Tukey outlier rejection.
    std::vector<size_t> outlier_isel =
      simple_tukey(x_resid_sel, y_resid_sel, phi_resid_sel);
    // get indices of good, then loop over good indics from first.
    std::vector<bool> good_sel(x_resid_sel.size(), true);
    for (const size_t& isel : outlier_isel) {
        good_sel[isel] = false;
    }
    std::vector<std::size_t> final_sel{};
    for (int i = 0; i < good_sel.size(); i++) {
        if (good_sel[i]) {
            final_sel.push_back(sel[i]);
        }
    }
    // Do the selection and update the flags.
    ReflectionTable subrefls = reflections.select(final_sel);
    
    auto subflags_ = subrefls.column<std::size_t>("flags");
    auto& subflags = subflags_.value();

    for (int i=0; i<subflags.extent(0);++i){
        subflags(i,0) |= used_in_refinement_value;
        subflags(i,0) &= ~centroid_outlier_value;  // necessary?
    }

    return subrefls;
}

/**
 * @brief An initial filter of the reflection table
 * @param reflections The input reflection table.
 * @param hkl The input miller indices.
 * @param gonio The goniometer model.
 * @param beam The beam model.
 * @param close_to_spindle_cutoff The cutoff threshold for removing reflection for being close to the rotation axis.
 * @returns A subset of the input reflection table.
 */
ReflectionTable initial_filter(const ReflectionTable& reflections,
                               const mdspan_type<int>& hkl,
                               const Goniometer gonio,
                               const MonochromaticBeam beam,
                               const double close_to_spindle_cutoff) {
    auto flags_ = reflections.column<std::size_t>("flags");
    const auto& flags = flags_.value();
    auto s1_ = reflections.column<double>("s1");
    const auto& s1 = s1_.value();
    auto xyzobs_ = reflections.column<double>("xyzobs_mm");
    const auto& xyzobs = xyzobs_.value();
    Vector3d axis = gonio.get_rotation_axis();
    Vector3d s0 = beam.get_s0();

    // First select reflections that are not overloaded, have a valid hkl and
    // are not too close to the rotation axis.
    std::vector<bool> sel(flags.extent(0), true);
    for (int i = 0; i < sel.size(); i++) {
        if ((flags(i,0) & overloaded_value) == overloaded_value) {
            sel[i] = false;
        } else if (hkl(i,0) == 0 && hkl(i,1) == 0 && hkl(i,2) == 0) {
            sel[i] = false;
        } else if (std::abs(Eigen::Map<Vector3d>(&s1(i,0)).cross(s0).dot(axis)) <= close_to_spindle_cutoff) {
            sel[i] = false;
        }
        /*else if (!((*experiment.get_scan()).is_angle_valid(xyzobs[i][2]))){
            sel[i] = false;
        }*/
    }
    // Filter the table and the miller indices.
    ReflectionTable subrefls = reflections.select(sel);
    std::vector<int> filtered_hkl;
    for (int i = 0; i < sel.size(); i++) {
        if (sel[i]) {
            filtered_hkl.push_back(hkl(i,0));
            filtered_hkl.push_back(hkl(i,1));
            filtered_hkl.push_back(hkl(i,2));
        }
    }
    subrefls.add_column<int>("miller_index", filtered_hkl.size() /3, 3, filtered_hkl);
    auto subflags_ = subrefls.column<std::size_t>("flags");
    auto& subflags = subflags_.value();
    for (int i=0; i<subflags.extent(0);++i){
        subflags(i,0) &= ~used_in_refinement_value;  //unset the flag
    }

    return subrefls;
}

/**
 * @brief Select a subset of a specified suze from a reflection table
 * @param obs The input reflection table.
 * @param nref_per_degree The number of reflections per degree to select.
 * @param scan_width_degrees The extent of the scan, in degrees.
 * @param min_sample_size The minimum sample size to return.
 * @param max_sample_size The maximum sample size to return.
 * @returns A subset of the input reflection table.
 */
std::optional<ReflectionTable> select_sample(ReflectionTable& obs,
                              int nref_per_degree,
                              double scan_width_degrees,
                              int min_sample_size,
                              int max_sample_size) {
    auto flags_ = obs.column<std::size_t>("flags");
    auto& flags = flags_.value();
    int nrefs = flags.size();
    int sample_size =
      (nref_per_degree * std::max(std::round(scan_width_degrees), 1.0)) / 1;
    sample_size = std::max(sample_size, min_sample_size);
    if (max_sample_size) {
        sample_size = std::min(sample_size, max_sample_size);
    }
    if (sample_size < nrefs) {
        std::vector<std::size_t> sel = random_selection(nrefs, sample_size);
        ReflectionTable sel_obs = obs.select(sel);
        return std::move(sel_obs);
    }
    else {
        return {};
    }
    
}

/**
 * @brief Perform filtering before refinement, as done in the dials.refine reflection manager.
 * @param obs The input reflection table.
 * @param miller_indices The miller indices assigned to the data.
 * @param gonio The goniometer model.
 * @param crystal The crystal model.
 * @param beam The beam model.
 * @param panel The panel from the detector model.
 * @param scan_width_degrees The extent of the scan, in degrees.
 * @param n_ref_per_degree The number of reflections per degree to select.
 * @param close_to_spindle_cutoff A rejection threshold for reflections close to the rotation axis.
 * @param min_sample_size The minimum sample size to return.
 * @param max_sample_size The maximum sample size to return.
 * @returns A filtered subset of the input reflection table.
 */
ReflectionTable reflection_filter_preevaluation(
  const ReflectionTable& obs,
  const mdspan_type<int>& miller_indices,
  const Goniometer& gonio,
  const Crystal& crystal,
  const MonochromaticBeam& beam,
  const Panel& panel,
  double scan_width_degrees,
  int n_ref_per_degree = 100,
  double close_to_spindle_cutoff = 0.02,
  int min_sample_size = 1000,
  int max_sample_size = 0) {
    // First do an initial filter
    ReflectionTable filter_obs =
      initial_filter(obs, miller_indices, gonio, beam, close_to_spindle_cutoff);
    Matrix3d UB = crystal.get_A_matrix();
    // Predict the location of the reflections.
    simple_reflection_predictor(beam, gonio, UB, panel, filter_obs);
    // Do some outlier rejection and select a suitable subset.
    ReflectionTable outlier_filter_obs = outlier_filter(filter_obs);
    auto result = select_sample(outlier_filter_obs,
                                            n_ref_per_degree,
                                            scan_width_degrees,
                                            min_sample_size,
                                            max_sample_size);
    if (result.has_value()){
        ReflectionTable& sel_obs = result.value();
        return std::move(sel_obs);
    }                                 
    return outlier_filter_obs;
}
