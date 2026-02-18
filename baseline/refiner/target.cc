#pragma once

#include <cmath>
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <iostream>

#include "beam_parameterisation.cc"
#include "cell_parameterisation.cc"
#include "detector_parameterisation.cc"
#include "gradients_calculator.cc"
#include "orientation_parameterisation.cc"
#include "scan_static_predictor.cc"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

class Target {
  public:
    Target(Crystal &crystal,
           const Goniometer &goniometer,
           MonochromaticBeam &beam,
           Panel &panel,
           ReflectionTable &obs);
    std::vector<std::vector<double>> gradients() const;
    std::vector<double> residuals(std::vector<double> x);
    int nref() const;
    int nparams() const;
    CellParameterisation cell_parameterisation() const;
    OrientationParameterisation orientation_parameterisation() const;
    DetectorParameterisation detector_parameterisation() const;
    BeamParameterisation beam_parameterisation() const;
    std::vector<double> rmsds() const;

  private:
    Crystal crystal;
    Goniometer goniometer;
    MonochromaticBeam beam;
    Panel &panel_;
    ReflectionTable &obs;
    CellParameterisation cellparam;
    OrientationParameterisation orientationparam;
    DetectorParameterisation detectorparam;
    BeamParameterisation beamparam;
    GradientsCalculator calculator;
    int n_ref;
    int n_params;
    std::vector<double> rmsds_ = {0.0, 0.0, 0.0};
};

Target::Target(Crystal &crystal,
               const Goniometer &goniometer,
               MonochromaticBeam &beam,
               Panel &panel,
               ReflectionTable &obs)
    : crystal(crystal),
      goniometer(goniometer),
      beam(beam),
      panel_(panel),
      obs(obs),
      cellparam(crystal),
      orientationparam(crystal),
      detectorparam(panel_),
      beamparam(beam, goniometer),
      calculator(orientationparam, cellparam, goniometer, beamparam, detectorparam) {
    auto s1_ = obs.column<double>("s1");
    const auto &s1 = s1_.value();
    n_ref = s1.extent(0);
    n_params = beamparam.get_params().size() + orientationparam.get_params().size()
               + cellparam.get_params().size() + detectorparam.get_params().size();
};

int Target::nref() const {
    return n_ref;
}

int Target::nparams() const {
    return n_params;
}

std::vector<double> Target::rmsds() const {
    return rmsds_;
}

CellParameterisation Target::cell_parameterisation() const {
    return cellparam;
}

OrientationParameterisation Target::orientation_parameterisation() const {
    return orientationparam;
}

DetectorParameterisation Target::detector_parameterisation() const {
    return detectorparam;
}

BeamParameterisation Target::beam_parameterisation() const {
    return beamparam;
}

std::vector<double> Target::residuals(std::vector<double> x) {
    std::vector<double> beam_params = {x[0], x[1], x[2]};
    std::vector<double> U_params = {x[3], x[4], x[5]};
    std::vector<double> B_params = {x[6], x[7], x[8], x[9], x[10], x[11]};
    std::vector<double> detector_params = {x[12], x[13], x[14], x[15], x[16], x[17]};
    beamparam.set_params(beam_params);
    orientationparam.set_params(U_params);
    cellparam.set_params(B_params);
    detectorparam.set_params(detector_params);
    Matrix3d d = detectorparam.get_state();
    panel_.update(d);
    Vector3d s0 = beamparam.get_state();
    beam.set_s0(s0);
    Matrix3d B = cellparam.get_state();
    Matrix3d A = orientationparam.get_state() * B;
    simple_reflection_predictor(beam, goniometer, A, panel_, obs);
    auto xyzobs_ = obs.column<double>("xyzobs.mm.value");
    const auto &xyzobs_mm = xyzobs_.value();
    auto xyzcal_ = obs.column<double>("xyzcal.mm");
    const auto &xyzcal_mm = xyzcal_.value();
    std::vector<double> residuals(
      xyzobs_mm.size());  // residuals vector is all dx then dy then dz
    int n = xyzobs_mm.extent(0);
    double xsum = 0;
    double ysum = 0;
    double zsum = 0;
    for (int i = 0; i < n; ++i) {
        residuals[i] = xyzcal_mm(i, 0) - xyzobs_mm(i, 0);
        xsum += std::pow(residuals[i], 2);
    }
    for (int i = n, k = 0; i < (2 * n); ++i, ++k) {
        residuals[i] = xyzcal_mm(k, 1) - xyzobs_mm(k, 1);
        ysum += std::pow(residuals[i], 2);
    }
    for (int i = (2 * n), k = 0; i < (3 * n); ++i, ++k) {
        residuals[i] = xyzcal_mm(k, 2) - xyzobs_mm(k, 2);
        zsum += std::pow(residuals[i], 2);
    }
    rmsds_[0] = std::sqrt(xsum / n);
    rmsds_[1] = std::sqrt(ysum / n);
    rmsds_[2] = std::sqrt(zsum / n);
    return residuals;
}

std::vector<std::vector<double>> Target::gradients() const {
    return calculator.get_gradients(obs);
}