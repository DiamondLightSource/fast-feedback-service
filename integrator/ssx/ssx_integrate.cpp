#include "mosaicity_parameterisation.hpp"
#include "integrator/sigma_estimation.hpp"
#include "reflection_likelihood.hpp"
#include "fisher_scoring_max_likelihood.hpp"
#include "max_likelihood_target.hpp"
#include <cmath>
#include <math/math_utils.cuh>
#include <vector>

using Matrix3d = Eigen::Matrix3d;
using Vector2d = Eigen::Vector2d;

Vector3d ssx_integrate(const std::vector<Vector3d>& xyzcal_px,
    const std::vector<Vector3d>& xyzobs_px,
    const std::vector<Vector3d>& covariances,
    const std::vector<double>& intensities,
    const std::vector<Eigen::Vector3i>& miller_indices,
    const std::vector<Vector2d>& mobs,
    const Vector3d &s0,
    const Panel &panel,
    const Matrix3d &A
    ){
    double max_separation=2.0;
    // perform max separation filter
    std::vector<std::size_t> keep;
    for (int i=0;i<xyzcal_px.size();i++){
        if (std::pow(
            std::pow(xyzcal_px[i][0] - xyzobs_px[i][0],2) +
            std::pow(xyzcal_px[i][1] - xyzobs_px[i][1],2), 0.5) < max_separation){
                keep.push_back(i);
            }
    }
    // Now make filtered arrays - redo this to be smart and do nothing if no data filtered out?
    std::vector<double> intensities_f;
    std::vector<Vector3d> covariances_f;
    std::vector<Vector3d> xyzobs_f;
    std::vector<Vector3d> xyzcal_f;
    std::vector<Eigen::Vector3i> miller_indices_f;
    std::vector<Vector2d> mobs_f;

    intensities_f.reserve(keep.size());
    covariances_f.reserve(keep.size());
    xyzobs_f.reserve(keep.size());
    xyzcal_f.reserve(keep.size());
    miller_indices_f.reserve(keep.size());
    mobs_f.reserve(keep.size());

    for (auto i : keep) {
        intensities_f.push_back(intensities[i]);
        covariances_f.push_back(covariances[i]);
        xyzobs_f.push_back(xyzobs_px[i]);
        xyzcal_f.push_back(xyzcal_px[i]);
        miller_indices_f.push_back(miller_indices[i]);
        mobs_f.push_back(mobs[i]);
    }
    // End of filtering section.


    double tot_sigma_b = 0.0;
    int n = xyzcal_f.size();
    for (int i=0;i<n;i++){
        tot_sigma_b += (covariances_f[i][0] + covariances_f[i][1])/2.0;
    }
    double sigma_b_spot = std::pow(tot_sigma_b / n, 0.5);
    double sigma_b_rmsd = estimate_sigmab_2d(xyzcal_f, xyzobs_f, s0, panel);
    double overall_sigma_b = std::pow(std::pow(sigma_b_rmsd, 2) + std::pow(sigma_b_spot,2), 0.5);
    // for the sigma6 mosaicity model, sigma_b is used as the starting point for the diagonal terms
    // in the matrix

    // Note model must outlive max likelihood target due to reference.
    Simple6MosaicityParameterisation model = Simple6MosaicityParameterisation::from_sigma_d(overall_sigma_b);
    double s0_length = s0.norm();
    const std::size_t n1 = miller_indices_f.size();
    std::vector<Vector3d> sp_list;
    for (std::size_t i = 0; i < n1; ++i) {
        auto [xmm, ymm] = panel.px_to_mm(xyzobs_f[i][0], xyzobs_f[i][1]);
        Vector3d sp_i = panel.get_lab_coord(xmm, ymm);
        sp_i.normalize();
        sp_i = sp_i * s0_length;
        sp_list.push_back(sp_i);
    }

    MaximumLikelihoodTarget target(
        model,
        A,
        s0,
        sp_list,
        covariances_f,
        intensities_f,
        miller_indices_f,
        mobs_f
    );
    FisherScoringMaximumLikelihood scorer = FisherScoringMaximumLikelihood(model, target);
    scorer.solve();
    Matrix3d sigma = model.sigma();
    model.print_mosaicity();

    // now predict
    gemmi::SpaceGroup space_group = *gemmi::find_spacegroup_by_name("P1");
    gemmi::GroupOps crystal_symmetry_operations = space_group.operations();

    // Make detector from panel
    std::vector<Panel> panels;
    panels.push_back(panel);
    const Detector detector(panels);

    // For now return the mosaicity values for testing
    auto m = model.mosaicity();
    Vector3d m_vals = {m.min, m.mid, m.max};
    return m_vals;
    
    //predicted_data_stills results = predict_still(sigma, s0, detector, A, crystal_symmetry_operations);
}