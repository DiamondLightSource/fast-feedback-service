#ifndef DIALS_STATIC_PREDICTOR
#define DIALS_STATIC_PREDICTOR
#include <Eigen/Dense>
#include <dx2/detector.h>
#include <dx2/goniometer.h>
#include <dx2/beam.h>
#include <cmath>
#include <cassert>
#include <iostream>
#include "reflection_data.h"
const double two_pi = 2 * M_PI;

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

inline double mod2pi(double angle) {
  // E.g. treat 359.9999999 as 360
  if (std::abs(angle - two_pi) <= 1e-7) {
    angle = two_pi;
  }
  return angle - two_pi * floor(angle / two_pi);
}

Vector3d unit_rotate_around_origin(Vector3d vec, Vector3d unit, double angle){
    double cosang = std::cos(angle);
    Vector3d res = vec*cosang + (unit*(unit.dot(vec)) * (1.0-cosang)) + (unit.cross(vec)*std::sin(angle));
    return res;
}

// actually a repredictor, assumes all successful.
void simple_reflection_predictor(
  const MonochromaticBeam beam,
  const Goniometer gonio,
  //const Vector3d s0,//beam s0
  //const Matrix3d F,//fixed rot
  //const Matrix3d S,//setting rot
  //const Vector3d R, //get_rotation_axis_datum
  const Matrix3d UB,
  const Panel &panel,
  reflection_data &reflections
  //const int image_range_start,
  //const double osc_start,
  //const double osc_width
){
  std::vector<Vector3d> s1 = reflections.s1;
  const std::vector<Vector3d> xyzobs_mm = reflections.xyzobs_mm;
  const std::vector<bool> entering = reflections.entering;
  std::vector<std::size_t> flags = reflections.flags;
  std::vector<Vector3d> xyzcal_mm = reflections.xyzcal_mm;
  const std::vector<Vector3i> hkl = reflections.miller_indices;
  // these setup bits are the same for all refls.
  Vector3d s0 = beam.get_s0();
  Matrix3d F = gonio.get_sample_rotation();//fixed rot
  Matrix3d S = gonio.get_setting_rotation();//setting rot
  Vector3d R = gonio.get_rotation_axis();
  Vector3d s0_ = S.inverse() * s0;
  Matrix3d FUB = F * UB;
  Vector3d m2 = R / R.norm();
  //Vector3d m2 = R.normalize();//fixed during refine
  Vector3d s0_m2_plane = s0.cross(S * R);
  s0_m2_plane.normalize();

  Vector3d m1 = m2.cross(s0_);
  m1.normalize(); //vary with s0
  Vector3d m3 = m1.cross(m2);
  m3.normalize(); //vary with s0
  double s0_d_m2 = s0_.dot(m2);
  double s0_d_m3 = s0_.dot(m3);

  /*std::vector<Vector3d> xyzcalmm = obs["xyzcal.mm"];
  std::vector<Vector3d> s1_all = obs["s1"];
  std::vector<bool> entering = obs["entering"];
  std::vector<Vector3d> xyzobs = obs["xyzobs.mm.value"];
  std::vector<Vector3i> hkl = obs["miller_index"];
  std::vector<std::size_t> flags = obs["flags"];*/
  size_t predicted_value = (1 << 0); //predicted flag
  // now call predict_rays with h and UB for a given refl
  for (int i=0;i<hkl.size();i++){
    const Vector3i h = hkl[i];
    const Vector3d hf{(double)h[0], (double)h[1], (double)h[2]};
    bool entering_i = entering[i];

    Vector3d pstar0 = FUB * hf;
    double pstar0_len_sq = pstar0.squaredNorm();
    if (pstar0_len_sq > 4 * s0_.squaredNorm()){
      flags[i] = flags[i] & ~predicted_value;
      continue;
    }
    double pstar0_d_m1 = pstar0.dot(m1);
    double pstar0_d_m2 = pstar0.dot(m2);
    double pstar0_d_m3 = pstar0.dot(m3);
    double pstar_d_m3 = (-(0.5 * pstar0_len_sq) - (pstar0_d_m2 * s0_d_m2)) / s0_d_m3;
    double rho_sq = (pstar0_len_sq - (pstar0_d_m2*pstar0_d_m2));
    double psq = pstar_d_m3*pstar_d_m3;
    if (rho_sq < psq){
      flags[i] = flags[i] & ~predicted_value;
      continue;
    }
    //DIALS_ASSERT(rho_sq >= sqr(pstar_d_m3));
    double pstar_d_m1 = sqrt(rho_sq - (psq));
    double p1 = pstar_d_m1 * pstar0_d_m1;
    double p2 = pstar_d_m3 * pstar0_d_m3;
    double p3 = pstar_d_m1 * pstar0_d_m3;
    double p4 = pstar_d_m3 * pstar0_d_m1;
    
    double cosphi1 = p1 + p2;
    double sinphi1 = p3 - p4;
    double a1 = atan2(sinphi1, cosphi1);
    // ASSERT must be in range? is_angle_in_range

    // check each angle
    Vector3d pstar = S * unit_rotate_around_origin(pstar0, m2, a1);
    Vector3d s1_this = s0_ + pstar;
    bool this_entering = s1_this.dot(s0_m2_plane) < 0.;
    double angle;
    if (this_entering == entering_i){
      // use this s1 and a1 (mod 2pi)
      angle = mod2pi(a1);
    }
    else {
      double cosphi2 = -p1 + p2;
      double sinphi2 = -p3 - p4;
      double a2 = atan2(sinphi2, cosphi2);
      pstar = S * unit_rotate_around_origin(pstar0, m2, a2);
      s1_this = s0_ + pstar;
      this_entering = s1_this.dot(s0_m2_plane) < 0.;
      assert(this_entering == entering_i);
      angle = mod2pi(a2);
    }

    // only need frame if calculating xyzcalpx, but not needed for evaluation
    //double frame = image_range_start + ((angle - osc_start) / osc_width) - 1;
    //Vector3d v = D * s1;
    std::array<double, 2> mm = panel.get_ray_intersection(s1_this); //v = D * s1; v[0]/v[2], v[1]/v[2]
    //scitbx::vec2<double> px = Detector[0].millimeter_to_pixel(mm); // requires call to parallax corr
    
    // match full turns
    double phiobs = xyzobs_mm[i][2];
    // first fmod positive
    double val = std::fmod(phiobs, two_pi);
    while (val < 0) val += two_pi;
    double resid = angle - val;
    // second fmod positive
    double val2 = std::fmod(resid+M_PI, two_pi);
    while (val2 < 0) val2 += two_pi;
    val2 -= M_PI;
    
    xyzcal_mm[i] = {mm[0], mm[1], phiobs + val2};
    //xyzcalpx[i] = {px[0], px[1], frame};
    s1[i] = s1_this;
    flags[i] = flags[i] | predicted_value;
  }
  reflections.flags = flags;
  reflections.xyzcal_mm = xyzcal_mm;
}

#endif  // DIALS_STATIC_PREDICTOR