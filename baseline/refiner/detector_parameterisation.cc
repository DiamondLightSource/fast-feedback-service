#ifndef DIALS_RESEARCH_DPARAM
#define DIALS_RESEARCH_DPARAM
#include <dx2/detector.hpp>
#include <Eigen/Dense>
#include "refinement_utils.cc"
#include <cmath>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

class SimpleDetectorParameterisation {
public:
  SimpleDetectorParameterisation(
    const Panel &panel,
    bool fix_dist,
    bool fix_shift1,
    bool fix_shift2,
    bool fix_tau1,
    bool fix_tau2,
    bool fix_tau3);
  std::vector<double> get_params() const;
  void set_params(std::vector<double> p);
  Matrix3d get_state() const;
  std::vector<Matrix3d> get_dS_dp() const;
  bool dist_fixed() const;
  bool shift1_fixed() const;
  bool shift2_fixed() const;
  bool tau1_fixed() const;
  bool tau2_fixed() const;
  bool tau3_fixed() const;

private:
  std::vector<double> params_ = {0.0,0.0,0.0,0.0,0.0,0.0}; //
  void compose();
  std::vector<Matrix3d> dS_dp{
    6,
    Matrix3d {{0, 0, 0}, {0, 0, 0}, {0,0,0}}};
  Vector3d initial_offset{{0.0,0.0,0.0}};
  Vector3d initial_d1{{0.0,0.0,0.0}};
  Vector3d initial_d2{{0.0,0.0,0.0}};
  Vector3d initial_dn{{0.0,0.0,0.0}};
  Vector3d initial_origin{{0.0,0.0,0.0}};
  Vector3d current_origin{{0.0,0.0,0.0}};
  Vector3d current_d1{{0.0,0.0,0.0}};
  Vector3d current_d2{{0.0,0.0,0.0}};
  bool _fix_dist{true};
  bool _fix_shift1{false};
  bool _fix_shift2{true};
  bool _fix_tau1{true};
  bool _fix_tau2{true};
  bool _fix_tau3{true};
};

void SimpleDetectorParameterisation::compose(){
    double t1r = params_[3] / 1000.0;
    double t2r = params_[4] / 1000.0;
    double t3r = params_[5] / 1000.0;
    Matrix3d Tau1 = axis_and_angle_as_rot(initial_dn, t1r);
    Matrix3d dTau1_dtau1 = dR_from_axis_and_angle(initial_dn, t1r);
    Matrix3d Tau2 = axis_and_angle_as_rot(initial_d1, t2r);
    Matrix3d dTau2_dtau2 = dR_from_axis_and_angle(initial_d1, t2r);
    Matrix3d Tau3 = axis_and_angle_as_rot(initial_d2, t3r);
    Matrix3d dTau3_dtau3 = dR_from_axis_and_angle(initial_d2, t3r);
    Matrix3d Tau32 = Tau3 * Tau2;
    Matrix3d Tau321 = Tau32 * Tau1;
    Vector3d P0 = params_[0] * initial_dn;
    Vector3d Px = P0 + initial_d1;
    Vector3d Py = P0 + initial_d2;
    Vector3d dsv = P0 + (params_[1] * initial_d1) + (params_[2] * initial_d2);
    Vector3d dorg = (Tau321 * dsv) - (Tau32 * P0) + P0;
    Vector3d d1 = (Tau321 * (Px-P0));
    d1.normalize();
    Vector3d d2 = (Tau321 * (Py-P0));
    d2.normalize();
    Vector3d dn = d1.cross(d2);
    dn.normalize();
    current_d1 = d1;
    current_d2 = d2;

    //Vector3d d2 = dn.cross(d1);
    Vector3d o = dorg + initial_offset[0] * d1 + initial_offset[1] * d2;
    //new_state = {"d1": d1, "d2": d2, "origin": o}
    current_origin = o;

    // derivative of dorg wrt dist
    Vector3d ddorg_ddist = (Tau321*initial_dn) - (Tau32*initial_dn) + initial_dn;
    Vector3d ddorg_dshift1 = Tau321 * initial_d1;
    Vector3d ddorg_dshift2 = Tau321 * initial_d2;
    // derivative wrt tau1, tau2, tau3
    Matrix3d dTau321_dtau1 = Tau32 * dTau1_dtau1;
    Vector3d ddorg_dtau1 = dTau321_dtau1 * dsv;
    Matrix3d dTau32_dtau2 = Tau3 * dTau2_dtau2;
    Matrix3d dTau321_dtau2 = dTau32_dtau2 * Tau1;
    Vector3d ddorg_dtau2 = dTau321_dtau2 * dsv - dTau32_dtau2 * P0;
    Matrix3d dTau32_dtau3 = dTau3_dtau3 * Tau2;
    Matrix3d dTau321_dtau3 = dTau32_dtau3 * Tau1;
    Vector3d ddorg_dtau3 = dTau321_dtau3 * dsv - dTau32_dtau3 * P0;

    // Now derivatives of the direction d1, where d1 = (Tau321 * (Px - P0)).normalize()
    //Vector3d dd1_ddist{0.0, 0.0, 0.0};
    //Vector3d dd1_dshift1{0.0, 0.0, 0.0};
    //Vector3d dd1_dshift2{0.0, 0.0, 0.0};
    Vector3d dd1_dtau1 = dTau321_dtau1 * (Px - P0);
    Vector3d dd1_dtau2 = dTau321_dtau2 * (Px - P0);
    Vector3d dd1_dtau3 = dTau321_dtau3 * (Px - P0);

    // Derivatives of the direction d2, where d2 = (Tau321 * (Py - P0)).normalize()
    //Vector3d dd2_ddist{0.0, 0.0, 0.0};
    //Vector3d dd2_dshift1{0.0, 0.0, 0.0};
    //Vector3d dd2_dshift2{0.0, 0.0, 0.0};
    Vector3d dd2_dtau1 = dTau321_dtau1 * (Py - P0);
    Vector3d dd2_dtau2 = dTau321_dtau2 * (Py - P0);
    Vector3d dd2_dtau3 = dTau321_dtau3 * (Py - P0);

    // Derivatives of the direction dn, where dn = d1.cross(d2).normalize()
    // These derivatives are not used
    Vector3d do_ddist = ddorg_ddist;// + initial_offset[0] * dd1_ddist + initial_offset[1] * dd2_ddist;
    Vector3d do_dshift1 = ddorg_dshift1;// + initial_offset[0] * dd1_dshift1 + initial_offset[1] * dd2_dshift1;
    Vector3d do_dshift2 = ddorg_dshift2;// + initial_offset[0] * dd1_dshift2 + initial_offset[1] * dd2_dshift2;
    Vector3d do_dtau1 = ddorg_dtau1 + initial_offset[0] * dd1_dtau1 + initial_offset[1] * dd2_dtau1;
    Vector3d do_dtau2 = ddorg_dtau2 + initial_offset[0] * dd1_dtau2 + initial_offset[1] * dd2_dtau2;
    Vector3d do_dtau3 = ddorg_dtau3 + initial_offset[0] * dd1_dtau3 + initial_offset[1] * dd2_dtau3;
    do_dtau1 /= 1000.0;
    do_dtau2 /= 1000.0;
    do_dtau3 /= 1000.0;
    dd1_dtau1 /= 1000.0;
    dd1_dtau2 /= 1000.0;
    dd1_dtau3 /= 1000.0;
    dd2_dtau1 /= 1000.0;
    dd2_dtau2 /= 1000.0;
    dd2_dtau3 /= 1000.0;

    dS_dp[0] = Matrix3d{
        {0.0,0.0,0.0},//dd1_ddist[0], dd1_ddist[1], dd1_ddist[2],
        {0.0,0.0,0.0},//dd2_ddist[0], dd2_ddist[1], dd2_ddist[2],
        {do_ddist[0], do_ddist[1], do_ddist[2]}
    }.transpose();
    dS_dp[1] = Matrix3d{
        {0.0,0.0,0.0},//dd1_dshift1[0], dd1_dshift1[1], dd1_dshift1[2],
        {0.0,0.0,0.0},//dd2_dshift1[0], dd2_dshift1[1], dd2_dshift1[2],
        {do_dshift1[0], do_dshift1[1], do_dshift1[2]}
    }.transpose();
    dS_dp[2] = Matrix3d{
        {0.0,0.0,0.0},//dd1_dshift2[0], dd1_dshift2[1], dd1_dshift2[2],
        {0.0,0.0,0.0},//dd2_dshift2[0], dd2_dshift2[1], dd2_dshift2[2],
        {do_dshift2[0], do_dshift2[1], do_dshift2[2]}
    }.transpose();

    dS_dp[3] = Matrix3d{
        {dd1_dtau1[0], dd1_dtau1[1], dd1_dtau1[2]},
        {dd2_dtau1[0], dd2_dtau1[1], dd2_dtau1[2]},
        {do_dtau1[0], do_dtau1[1], do_dtau1[2]}
    }.transpose();
    dS_dp[4] = Matrix3d{
        {dd1_dtau2[0], dd1_dtau2[1], dd1_dtau2[2]},
        {dd2_dtau2[0], dd2_dtau2[1], dd2_dtau2[2]},
        {do_dtau2[0], do_dtau2[1], do_dtau2[2]}
    }.transpose();
    dS_dp[5] = Matrix3d{
        {dd1_dtau3[0], dd1_dtau3[1], dd1_dtau3[2]},
        {dd2_dtau3[0], dd2_dtau3[1], dd2_dtau3[2]},
        {do_dtau3[0], do_dtau3[1], do_dtau3[2]}
    }.transpose();
}

SimpleDetectorParameterisation::SimpleDetectorParameterisation(
    const Panel& p,
    bool fix_dist=false,
    bool fix_shift1=false,
    bool fix_shift2=false,
    bool fix_tau1=false,
    bool fix_tau2=false,
    bool fix_tau3=false): 
        _fix_dist{fix_dist}, _fix_shift1{fix_shift1}, _fix_shift2{fix_shift2},
        _fix_tau1{fix_tau1}, _fix_tau2{fix_tau2}, _fix_tau3{fix_tau3}{
    //const dxtbx::model::Panel& p = Detector[0];
    Vector3d so = p.get_origin();
    Vector3d d1 = p.get_fast_axis();
    Vector3d d2 = p.get_slow_axis();
    Vector3d dn = p.get_normal();
    initial_d1 = d1;
    initial_d2 = d2;
    initial_dn = dn;
    double panel_lim_x = p.get_image_size_mm()[0];
    double panel_lim_y = p.get_image_size_mm()[1];
    initial_offset = {-0.5 * panel_lim_x, -0.5 * panel_lim_y, 0.0};
    Vector3d dorg = so - (initial_offset[0]*d1) - (initial_offset[1]*d2);
    params_[0] = p.get_directed_distance();
    Vector3d shift = dorg - dn*params_[0];
    params_[1] = shift.dot(d1);
    params_[2] = shift.dot(d2);
    params_[3] = 0.0;
    params_[4] = 0.0;
    params_[5] = 0.0;
    compose();
}

Matrix3d SimpleDetectorParameterisation::get_state() const{
    Matrix3d m{
        {current_d1[0], current_d2[0], current_origin[0]},
        {current_d1[1], current_d2[1], current_origin[1]},
        {current_d1[2], current_d2[2], current_origin[2]}
    };
    return m;
}

std::vector<double> SimpleDetectorParameterisation::get_params() const {
  return params_;
}
void SimpleDetectorParameterisation::set_params(std::vector<double> p) {
  params_ = p;
  compose();
}
std::vector<Matrix3d> SimpleDetectorParameterisation::get_dS_dp() const  {
  return dS_dp;
}

bool SimpleDetectorParameterisation::dist_fixed() const {
    return _fix_dist;
}
bool SimpleDetectorParameterisation::shift1_fixed() const {
    return _fix_shift1;
}
bool SimpleDetectorParameterisation::shift2_fixed() const {
    return _fix_shift2;
}
bool SimpleDetectorParameterisation::tau1_fixed() const {
    return _fix_tau1;
}
bool SimpleDetectorParameterisation::tau2_fixed() const {
    return _fix_tau2;
}
bool SimpleDetectorParameterisation::tau3_fixed() const {
    return _fix_tau3;
}

#endif  // DIALS_RESEARCH_DPARAM