#include <dx2/crystal.hpp>
#include <dx2/goniometer.hpp>
#include "detector_parameterisation.cc"
#include "beam_parameterisation.cc"
#include "orientation_parameterisation.cc"
#include "cell_parameterisation.cc"
#include <dx2/reflection.hpp>
#include "scan_static_predictor.cc"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

class GradientsCalculator {
public:
    GradientsCalculator(
        OrientationParameterisation &uparam,
        CellParameterisation &bparam,
        const Goniometer &goniometer,
        BeamParameterisation& beamparam,
        DetectorParameterisation& Dparam) ;
    std::vector<std::vector<double>> get_gradients(const ReflectionTable &obs) const;

private:
  OrientationParameterisation uparam;
  CellParameterisation bparam;
  Goniometer goniometer;
  BeamParameterisation beamparam;
  DetectorParameterisation& Dparam;
};

GradientsCalculator::GradientsCalculator(
    OrientationParameterisation& uparam,
    CellParameterisation& bparam,
    const Goniometer &goniometer,
    BeamParameterisation& beamparam,
    DetectorParameterisation& Dparam) :
uparam(uparam), bparam(bparam), goniometer(goniometer), beamparam(beamparam), Dparam(Dparam) {};

std::vector<std::vector<double>> GradientsCalculator::get_gradients(const ReflectionTable &obs) const {
    auto s1_ = obs.column<double>("s1");
    const auto& s1 = s1_.value();
    int n_ref = s1.extent(0);
    // assume one panel detector for now
    // Some templating to handle multi-panel/single panel detectors? will need a function to pass in a panel array
    // and return D array.
    Matrix3d state = Dparam.get_state();
    std::vector<Matrix3d> D(n_ref, state.inverse());
    Vector3d s0 = beamparam.get_state();
    Matrix3d B = bparam.get_state();
    Matrix3d U = uparam.get_state();

    Matrix3d S = goniometer.get_setting_rotation();
    Vector3d axis = goniometer.get_rotation_axis();
    Matrix3d F = goniometer.get_sample_rotation();
    Matrix3d UB = U * B;
    
    auto xyz_ = obs.column<double>("xyzcal.mm");
    const auto& xyz = xyz_.value();
    auto hkl_ = obs.column<int>("miller_index");
    const auto& hkl = hkl_.value();
    std::vector<Vector3d> r(n_ref);
    std::vector<Vector3d> pv(n_ref);
    std::vector<Vector3d> e_X_r(n_ref);
    std::vector<double> e_r_s0(n_ref);
    std::vector<double> w_inv(n_ref);
    std::vector<double> uw_inv(n_ref);
    std::vector<double> vw_inv(n_ref);
    for (int i=0;i<n_ref;i++){
        Eigen::Map<Vector3d> s1_i(&s1(i, 0));
        Vector3d pv_this = D[i] * s1_i;
        pv[i] = pv_this;
        double pvinv = 1.0 / pv_this[2];
        w_inv[i] = pvinv;
        uw_inv[i] = pvinv * pv_this[0];
        vw_inv[i] = pvinv * pv_this[1];
        Vector3d hkl_i = {
            static_cast<double>(hkl(i, 0)),
            static_cast<double>(hkl(i, 1)),
            static_cast<double>(hkl(i, 2))};
        Vector3d UBH = UB * hkl_i;
        Vector3d q = F * UBH;
        Vector3d r_i = S * unit_rotate_around_origin(q, axis, xyz(i,2));
        r[i] = r_i;
        e_X_r[i] = (S * axis).cross(r_i);
        e_r_s0[i] = e_X_r[i].dot(s0);
    }

    std::vector<std::vector<double>> gradients;
    std::vector<Matrix3d> ds_dp = uparam.get_dS_dp();
    std::vector<Matrix3d> db_dp = bparam.get_dS_dp();
    std::vector<Vector3d> dbeam_dp = beamparam.get_dS_dp();
    std::vector<Matrix3d> dD_dp = Dparam.get_dS_dp();

    //beam derivatives
    std::vector<bool> free;
    free.push_back(!beamparam.in_spindle_plane_fixed());
    free.push_back(!beamparam.out_spindle_plane_fixed());
    free.push_back(!beamparam.wavelength_fixed());
    for (int j=0;j<dbeam_dp.size();j++){
        // each gradient is dx/dp, dy/dp, dz/dp
        std::vector<double> gradient(n_ref*3);
        Vector3d dbeam_dp_j = dbeam_dp[j];
        if (free[j]){
            for (int k=0;k<n_ref;k++){
                double dphi = -1.0 * dbeam_dp_j.dot(r[k]) / e_r_s0[k];
                Vector3d dpv = D[k] * ((e_X_r[k] * dphi) + dbeam_dp_j);
                double w_inv_this = w_inv[k];
                gradient[k] = w_inv_this * (dpv[0] - dpv[2] * uw_inv[k]);
                gradient[k+n_ref] = w_inv_this * (dpv[1] - dpv[2] * vw_inv[k]);
                gradient[k+(2*n_ref)] = dphi;
            }
        }
        gradients.push_back(gradient);
    }
    
    for (int j=0;j<ds_dp.size();j++){
        // each gradient is dx/dp, dy/dp, dz/dp
        std::vector<double> gradient(n_ref*3);
        Matrix3d ds_dp_j = ds_dp[j];
        for (int k=0;k<n_ref;k++){
            Vector3d hkl_k = {
                static_cast<double>(hkl(k, 0)),
                static_cast<double>(hkl(k, 1)),
                static_cast<double>(hkl(k, 2))};
            Vector3d tmp = F * (ds_dp_j * (B * hkl_k));
            Vector3d dr = S * unit_rotate_around_origin(tmp, axis, xyz(k,2));
            Eigen::Map<Vector3d> s1_k(&s1(k, 0));
            double dphi = -1.0 * (dr.dot(s1_k)) / e_r_s0[k];
            Vector3d dpv = D[k] * (dr + e_X_r[k]*dphi);
            double w_inv_this = w_inv[k];
            gradient[k] = w_inv_this * (dpv[0] - dpv[2] * uw_inv[k]);
            gradient[k+n_ref] = w_inv_this * (dpv[1] - dpv[2] * vw_inv[k]);
            gradient[k+(2*n_ref)] = dphi;
        }
        gradients.push_back(gradient);
    }
    
    for (int j=0;j<db_dp.size();j++){
        // each gradient is dx/dp, dy/dp, dz/dp
        std::vector<double> gradient(n_ref*3);
        Matrix3d db_dp_j = db_dp[j];
        for (int k=0;k<n_ref;k++){
            Vector3d hkl_k = {
                static_cast<double>(hkl(k, 0)),
                static_cast<double>(hkl(k, 1)),
                static_cast<double>(hkl(k, 2))};
            Vector3d tmp = F * (U * (db_dp_j * hkl_k));
            Vector3d dr = S * unit_rotate_around_origin(tmp, axis, xyz(k,2));
            Eigen::Map<Vector3d> s1_k(&s1(k, 0));
            double dphi = -1.0 * (dr.dot(s1_k)) / e_r_s0[k];
            Vector3d dpv = D[k] * (dr + e_X_r[k]*dphi);
            double w_inv_this = w_inv[k];
            gradient[k] = w_inv_this * (dpv[0] - dpv[2] * uw_inv[k]);
            gradient[k+n_ref] = w_inv_this * (dpv[1] - dpv[2] * vw_inv[k]);
            gradient[k+(2*n_ref)] = dphi;
        }
        gradients.push_back(gradient);
    }

    for (int j=0;j<dD_dp.size();j++){
        // each gradient is dx/dp, dy/dp, dz/dp
        std::vector<double> gradient(n_ref*3);
        Matrix3d dD_dp_j = dD_dp[j];
        for (int k=0;k<n_ref;k++){
            Vector3d dpv = (D[k] * dD_dp_j * -1.0) * pv[k];
            double w_inv_this = w_inv[k];
            gradient[k] = w_inv_this * (dpv[0] - dpv[2] * uw_inv[k]);
            gradient[k+n_ref] = w_inv_this * (dpv[1] - dpv[2] * vw_inv[k]);
            // note there is no dphi component here.
        }
        gradients.push_back(gradient);
    }
    

    return gradients;
}
