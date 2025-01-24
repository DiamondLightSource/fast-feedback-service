#include <Eigen/Dense>
#include <iostream>
#include <random>
#include "scanstaticpredictor.cc"
#include <dx2/detector.h>
#include <dx2/beam.h>
#include <dx2/scan.h>
#include <dx2/crystal.h>
#include <dx2/goniometer.h>
#include "reflection_data.h"
// select on id before here.
using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::Vector3i;


std::vector<std::size_t> random_selection(int pop_size, int sample_size, int seed=43){
    std::mt19937 mt(seed);
    // get random selection up to pop size, then cut down to sample size
    // then sort
    std::vector<size_t> result(pop_size);
    std::vector<size_t>::iterator r = result.begin();
    for (size_t i=0;i<pop_size;i++){
        *r++ = i;
    }
    r = result.begin();
    for(std::size_t i=0;i<pop_size;i++) {
        std::size_t j = static_cast<std::size_t>(mt()) % pop_size;
        std::swap(r[i], r[j]);
    }
    result.resize(sample_size);
    std::sort(result.begin(), result.end());
    return result;
}





std::vector<size_t> simple_tukey(std::vector<double> xresid, std::vector<double> yresid, std::vector<double>phi_resid){
    std::vector<size_t> sel {};//(xresid.size(), true);
    double iqr_multiplier = 3.0;
    std::vector<double> xresid_unsorted(xresid.begin(), xresid.end());
    std::vector<double> yresid_unsorted(yresid.begin(), yresid.end());
    std::vector<double> phi_resid_unsorted(phi_resid.begin(), phi_resid.end());
    std::sort(xresid.begin(), xresid.end());
    std::sort(yresid.begin(), yresid.end());
    std::sort(phi_resid.begin(), phi_resid.end());

    // this is the way scitbx.math.five_number_summary does iqr
    int n_lower=0;
    double Q1x=0.0;
    double Q1y=0.0;
    double Q1p=0.0;
    double Q3x=0.0;
    double Q3y=0.0;
    double Q3p=0.0;
    int inc = 0; // an overlap offset if the number of items is odd.
    if (xresid.size() % 2){
        n_lower = (xresid.size() / 2) + 1;
        inc = -1;
    }
    else {
        n_lower = xresid.size() / 2;
    }
    if (n_lower % 2){
        // FIXME verify this branch correct
        Q1x = xresid[n_lower / 2];
        Q1y = yresid[n_lower / 2];
        Q1p = phi_resid[n_lower / 2];
        Q3x = xresid[n_lower + 1 + (n_lower/ 2)]; 
        Q3y = yresid[n_lower + 1 + (n_lower/ 2)];
        Q3p = phi_resid[n_lower + 1 + (n_lower/ 2)];
    }
    else {
        Q1x = (xresid[n_lower / 2] + xresid[(n_lower / 2) -1]) / 2.0;
        Q1y = (yresid[n_lower / 2] + yresid[(n_lower / 2) -1]) / 2.0;
        Q1p = (phi_resid[n_lower / 2] + phi_resid[(n_lower / 2) -1]) / 2.0;
        Q3x = (xresid[n_lower + inc+(n_lower / 2)] + xresid[n_lower +inc+(n_lower / 2) -1]) / 2.0;
        Q3y = (yresid[n_lower +inc+(n_lower / 2)] + yresid[n_lower +inc+(n_lower / 2) -1]) / 2.0;
        Q3p = (phi_resid[n_lower +inc+(n_lower / 2)] + phi_resid[n_lower +inc+(n_lower / 2) -1]) / 2.0;
    }
    double iqrx = Q3x - Q1x;
    double uppercutx = (iqrx * iqr_multiplier) + Q3x;
    double lowercutx = Q1x -(iqrx * iqr_multiplier);

    double iqry = Q3y - Q1y;
    double uppercuty = (iqry * iqr_multiplier) + Q3y;
    double lowercuty = Q1y -(iqry * iqr_multiplier);

    double iqrp = Q3p - Q1p;
    double uppercutp = (iqrp * iqr_multiplier) + Q3p;
    double lowercutp = Q1p -(iqrp * iqr_multiplier);

    for (int i=0;i<xresid_unsorted.size();i++){
        if (xresid_unsorted[i] > uppercutx){
            sel.push_back(i);
            //sel[i] = false;
        }
        else if (xresid_unsorted[i] < lowercutx){
            sel.push_back(i);
            //sel[i] = false;
        }
        else if (yresid_unsorted[i] > uppercuty){
            sel.push_back(i);
            //sel[i] = false;
        }
        else if (yresid_unsorted[i] < lowercuty){
            sel.push_back(i);
            //sel[i] = false;
        }
        else if (phi_resid_unsorted[i] > uppercutp){
            sel.push_back(i);
            //sel[i] = false;
        }
        else if (phi_resid_unsorted[i] < lowercutp){
            sel.push_back(i);
            //sel[i] = false;
        }
    }
    return sel;
}

reflection_data outlier_filter(reflection_data &reflections){
    // filter on predicted
    std::vector<std::size_t> flags = reflections.flags;
    std::vector<Vector3d> xyzobs = reflections.xyzobs_mm;
    std::vector<Vector3d> xyzcal = reflections.xyzcal_mm;
    std::vector<double> x_resid(reflections.flags.size(), 0.0);
    std::vector<double> y_resid(reflections.flags.size(), 0.0);
    std::vector<double> phi_resid(reflections.flags.size(), 0.0);
    std::vector<std::size_t> sel {};
    
    size_t predicted_value = (1 << 0); //predicted flag
    for (int i=0;i<flags.size();i++){
        if ((flags[i] & predicted_value) == predicted_value){
            Vector3d xyzobsi = xyzobs[i];
            Vector3d xyzcali = xyzcal[i];
            x_resid[i] = xyzcali[0] - xyzobsi[0];
            y_resid[i] = xyzcali[1] - xyzobsi[1];
            phi_resid[i] = xyzcali[2] - xyzobsi[2];
            sel.push_back(i);
        }
    }
    // avoid selecting on the whole table.
    std::vector<double> x_resid_sel(sel.size());
    std::vector<double> y_resid_sel(sel.size());
    std::vector<double> phi_resid_sel(sel.size());
    for (int i=0;i<sel.size();i++){
        int idx = sel[i];
        x_resid_sel[i] = x_resid[idx];
        y_resid_sel[i] = y_resid[idx];
        phi_resid_sel[i] = phi_resid[idx];
    }

    std::vector<size_t> outlier_isel = simple_tukey(x_resid_sel, y_resid_sel, phi_resid_sel);
    // get indices of good, then loop over good indics from first.
    std::vector<bool> good_sel(x_resid_sel.size(), true);
    for (int i=0;i<outlier_isel.size();i++){
        good_sel[outlier_isel[i]] = false;
    }
    std::vector<std::size_t> final_sel {}; 
    for (int i=0;i<good_sel.size();i++){
        if (good_sel[i]){
            final_sel.push_back(sel[i]);
        }
    }
    reflection_data subrefls = select(reflections, final_sel);
    std::vector<std::size_t> subflags = subrefls.flags;

    // unset centroid outlier flag (necessary?)
    // set used_in_refinement
    size_t used_in_refinement_value = (1 << 3); //used in refinement flag
    size_t centroid_outlier_value = (1 << 17); //ucentroid outlier flag
    for (int i=0;i<subrefls.flags.size();i++){
        subflags[i] |= used_in_refinement_value;
        subflags[i] &= ~centroid_outlier_value;
    }
    subrefls.flags = subflags;

    return subrefls;
}

reflection_data initial_refman_filter(
    reflection_data const& reflections,
    std::vector<Vector3i> const& hkl,
    Goniometer gonio, MonochromaticBeam beam,
    double close_to_spindle_cutoff){
    std::vector<std::size_t> flags = reflections.flags;
    //std::vector<Vector3i> hkl = miller_indices;
    std::vector<Vector3d> s1 = reflections.s1;
    //std::vector<int> id = reflections["id"];
    std::vector<Vector3d> xyzobs = reflections.xyzobs_mm;
    Vector3d axis = gonio.get_rotation_axis();
    Vector3d s0 = beam.get_s0();

    // get flags ('overloaded')
    size_t overloaded_value = (1 << 10); //overloaded flag
    std::vector<bool> sel(flags.size(), true);
    Vector3i null = {0,0,0};
    for (int i=0;i<sel.size();i++){
        /*if (id[i] != 0.0){
            sel[i] = false;
        }*/
        if ((flags[i] & overloaded_value) == overloaded_value){
            sel[i] = false;
        }
        else if (hkl[i] == null){
            sel[i] = false;
        }
        else if (std::abs(s1[i].cross(s0).dot(axis)) < close_to_spindle_cutoff){
            sel[i] = false;
        }
        /*else if (!((*experiment.get_scan()).is_angle_valid(xyzobs[i][2]))){
            sel[i] = false;
        }*/
    }
    reflection_data subrefls = select(reflections, sel);
    std::vector<Vector3i> filtered_hkl;
    for (int i=0;i<sel.size();i++){
        if (sel[i]){
            filtered_hkl.push_back(hkl[i]);
        }
    }
    subrefls.miller_indices = filtered_hkl;

    //dials::af::reflection_table subrefls = dials::af::boost_python::reflection_table_suite::select_rows_flags(
    //    reflections, sel.const_ref());
    // now calculate entering flags (needed for prediction) and frame numbers
    size_t used_in_refinement_value = (1 << 3); //used in refinement flag
    for (int i=0;i<flags.size();i++){
        flags[i] &= ~used_in_refinement_value; //unset the flag
    }

    std::vector<bool> enterings(subrefls.flags.size(), false);
    // calculate the entering column - Move to one of above loops?
    std::vector<Vector3d> s1_sub = subrefls.s1;
    Vector3d vec = s0.cross(axis);
    for (int i=0;i<subrefls.flags.size();i++){
        enterings[i] = (s1_sub[i].dot(vec) < 0.0);
    }
    subrefls.entering = enterings;
    subrefls.flags = flags;
    return subrefls;
}

reflection_data select_sample(
    reflection_data &obs,
    int nref_per_degree,
    double scan_width_degrees,
    int min_sample_size,
    int max_sample_size){
    int nrefs = obs.flags.size();
    int sample_size = (nref_per_degree * std::max(std::round(scan_width_degrees),1.0)) / 1;
    sample_size = std::max(sample_size,min_sample_size);
    if (max_sample_size){
        sample_size = std::min(sample_size, max_sample_size);
    }
    if (sample_size < nrefs){
        std::vector<std::size_t> sel = random_selection(nrefs, sample_size);
        reflection_data sel_obs = select(obs, sel);
        return sel_obs;
    }
    return obs;
}

reflection_data reflection_filter_preevaluation(
    reflection_data const& obs,
    std::vector<Vector3i> const& miller_indices,
    const Goniometer &gonio,
    const Crystal &crystal,
    const MonochromaticBeam &beam,
    const Panel &panel,
    double scan_width_degrees,
    int n_ref_per_degree=100,
    double close_to_spindle_cutoff=0.02,
    int min_sample_size=1000,
    int max_sample_size=0
    ){
    reflection_data filter_obs = initial_refman_filter(obs, miller_indices, gonio, beam, close_to_spindle_cutoff);
    Matrix3d UB = crystal.get_A_matrix();
    simple_reflection_predictor(
        beam,
        gonio,
        UB,
        panel,
        filter_obs
    );
    filter_obs = outlier_filter(filter_obs);
    reflection_data sel_obs = select_sample(filter_obs, n_ref_per_degree, scan_width_degrees, min_sample_size, max_sample_size);
    return sel_obs;
}
