#ifndef DIALS_RESEARCH_BEAMPARAM
#define DIALS_RESEARCH_BEAMPARAM
#include <dx2/beam.hpp>
#include <dx2/goniometer.hpp>
#include "refinement_utils.cc"
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

class SimpleBeamParameterisation {
public:
  SimpleBeamParameterisation(
    const MonochromaticBeam& beam, const Goniometer& goniometer,
    bool fix_in_spindle_plane, bool fix_out_spindle_plane, bool fix_wavelength);
  std::vector<double> get_params() const;
  void set_params(std::vector<double> p);
  Vector3d get_state() const;
  std::vector<Vector3d> get_dS_dp() const;
  bool in_spindle_plane_fixed() const;
  bool out_spindle_plane_fixed() const;
  bool wavelength_fixed() const;

private:
  std::vector<double> params_ = {0.0,0.0,0.0}; //mu1, mu2, nu
  void compose();
  Vector3d istate_s0{};
  Vector3d istate_pol_norm{};
  Vector3d s0{};
  Vector3d pn{};
  Vector3d s0_plane_dir1{};
  Vector3d s0_plane_dir2{};
  std::vector<Vector3d> dS_dp{
    3,
    Vector3d(0.0, 0, 0)};
  bool _fix_in_spindle_plane{true};
  bool _fix_out_spindle_plane{false};
  bool _fix_wavelength{true};
};

void SimpleBeamParameterisation::compose(){
    double mu1rad = params_[0] / 1000.0;
    double mu2rad = params_[1] / 1000.0;
    Matrix3d Mu1 = axis_and_angle_as_rot(s0_plane_dir1, mu1rad);
    Matrix3d Mu2 = axis_and_angle_as_rot(s0_plane_dir2, mu2rad);
    Matrix3d dMu1_dmu1 = dR_from_axis_and_angle(s0_plane_dir1, mu1rad);
    Matrix3d dMu2_dmu2 = dR_from_axis_and_angle(s0_plane_dir2, mu2rad);
    // compose new state
    Matrix3d Mu21 = Mu2 * Mu1;
    Vector3d s0_new_dir = (Mu21 * istate_s0);
    s0_new_dir.normalize();
    pn = (Mu21 * istate_pol_norm);
    pn.normalize();
    s0 = params_[2] * s0_new_dir;

    //    # calculate derivatives of the beam direction wrt angles:
    //    #  1) derivative wrt mu1
    Matrix3d dMu21_dmu1 = Mu2 * dMu1_dmu1;
    Vector3d ds0_new_dir_dmu1 = dMu21_dmu1 * istate_s0;

    //    #  2) derivative wrt mu2
    Matrix3d dMu21_dmu2 = dMu2_dmu2 * Mu1;
    Vector3d ds0_new_dir_dmu2 = dMu21_dmu2 * istate_s0;

    //# calculate derivatives of the attached beam vector, converting
    //# parameters back to mrad
    dS_dp[0] = ds0_new_dir_dmu1 * params_[2] / 1000.0;
    dS_dp[1] = ds0_new_dir_dmu2 * params_[2] / 1000.0;
    dS_dp[2] = s0_new_dir;
}

SimpleBeamParameterisation::SimpleBeamParameterisation(
    const MonochromaticBeam& beam,
    const Goniometer& goniometer,
    bool fix_in_spindle_plane=true,
    bool fix_out_spindle_plane=false,
    bool fix_wavelength=true):
        _fix_in_spindle_plane{fix_in_spindle_plane},
        _fix_out_spindle_plane{fix_out_spindle_plane},
        _fix_wavelength{fix_wavelength} {
    s0 = beam.get_s0();
    istate_s0 = s0 / s0.norm();
    istate_pol_norm = beam.get_polarization_normal();
    Vector3d spindle = goniometer.get_rotation_axis();
    s0_plane_dir2 = s0.cross(spindle);
    s0_plane_dir2.normalize(); // axis associated with mu2
    s0_plane_dir1 = s0_plane_dir2.cross(s0);
    s0_plane_dir1.normalize(); //axis associated with mu1
    params_[2] = s0.norm();
    compose();
}

Vector3d SimpleBeamParameterisation::get_state() const {
  return s0;
}
std::vector<double> SimpleBeamParameterisation::get_params() const {
  return params_;
}
void SimpleBeamParameterisation::set_params(std::vector<double> p) {
  params_ = p;
  compose();
}
std::vector<Vector3d> SimpleBeamParameterisation::get_dS_dp() const {
  return dS_dp;
}
bool SimpleBeamParameterisation::in_spindle_plane_fixed() const {
  return _fix_in_spindle_plane;
}
bool SimpleBeamParameterisation::out_spindle_plane_fixed() const{
  return _fix_out_spindle_plane;
}
bool SimpleBeamParameterisation::wavelength_fixed() const{
  return _fix_wavelength;
}

#endif  // DIALS_RESEARCH_BEAMPARAM