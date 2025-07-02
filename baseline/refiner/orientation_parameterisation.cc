#pragma once
#include <dx2/crystal.hpp>
#include <dx2/goniometer.hpp>
#include "refinement_utils.cc"
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;

class CrystalOrientationCompose {
  public:
    CrystalOrientationCompose(const Matrix3d &U0,
                              double phi1,
                              const Vector3d &phi1_axis,
                              double phi2,
                              const Vector3d &phi2_axis,
                              double phi3,
                              const Vector3d &phi3_axis) {
      // convert angles from mrad to radians
      phi1 /= 1000.;
      phi2 /= 1000.;
      phi3 /= 1000.;

      // compose rotation matrices and their first order derivatives
      Matrix3d Phi1 = axis_and_angle_as_matrix(phi1_axis, phi1);
      Matrix3d dPhi1_dphi1 = dR_from_axis_and_angle(phi1_axis, phi1);

      Matrix3d Phi2 = axis_and_angle_as_matrix(phi2_axis, phi2);
      Matrix3d dPhi2_dphi2 = dR_from_axis_and_angle(phi2_axis, phi2);

      Matrix3d Phi3 = axis_and_angle_as_matrix(phi3_axis, phi3);
      Matrix3d dPhi3_dphi3 = dR_from_axis_and_angle(phi3_axis, phi3);

      // compose new state
      Matrix3d Phi21 = Phi2 * Phi1;
      Matrix3d Phi321 = Phi3 * Phi21;
      U_ = Phi321 * U0;

      // calculate derivatives of the state wrt parameters
      dU_dphi1_ = (Phi3 * Phi2 * dPhi1_dphi1 * U0) / 1000.0;
      dU_dphi2_ = (Phi3 * dPhi2_dphi2 * Phi1 * U0) / 1000.0;
      dU_dphi3_ = (dPhi3_dphi3 * Phi21 * U0) / 1000.0;
    }

    Matrix3d U() const {
      return U_;
    }

    Matrix3d dU_dphi1() const {
      return dU_dphi1_;
    }

    Matrix3d dU_dphi2() const {
      return dU_dphi2_;
    }

    Matrix3d dU_dphi3() const {
      return dU_dphi3_;
    }

  private:
    Matrix3d U_;
    Matrix3d dU_dphi1_;
    Matrix3d dU_dphi2_;
    Matrix3d dU_dphi3_;
  };

class OrientationParameterisation {
public:
  OrientationParameterisation(const Crystal& crystal);
  std::vector<double> get_params() const;
  void set_params(std::vector<double> p);
  Matrix3d get_state() const;
  std::vector<Matrix3d> get_dS_dp() const;

private:
  std::vector<double> params = {0.0, 0.0, 0.0};
  std::vector<Vector3d> axes{3, Vector3d(1.0, 0.0, 0.0)};
  void compose();
  Matrix3d istate{};
  Matrix3d U_{};
  std::vector<Matrix3d> dS_dp{
    3,
    Matrix3d{{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}}};
};

void OrientationParameterisation::compose() {
  CrystalOrientationCompose coc(
    istate, params[0], axes[0], params[1], axes[1], params[2], axes[2]);
  U_ = coc.U();
  dS_dp[0] = coc.dU_dphi1();
  dS_dp[1] = coc.dU_dphi2();
  dS_dp[2] = coc.dU_dphi3();
}

OrientationParameterisation::OrientationParameterisation(const Crystal& crystal) {
  istate = crystal.get_U_matrix();
  axes[1] = Vector3d(0.0, 1.0, 0.0);
  axes[2] = Vector3d(0.0, 0.0, 1.0);
  compose();
}

std::vector<double> OrientationParameterisation::get_params() const {
  return params;
}
Matrix3d OrientationParameterisation::get_state() const {
  return U_;
}
void OrientationParameterisation::set_params(std::vector<double> p) {
  assert(p.size() == 3);
  params = p;
  compose();
}
std::vector<Matrix3d> OrientationParameterisation::get_dS_dp() const {
  return dS_dp;
}
