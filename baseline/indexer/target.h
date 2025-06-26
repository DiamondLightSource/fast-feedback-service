
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include "detector_parameterisation.h"
#include "beam_parameterisation.h"
#include "U_parameterisation.h"
#include "B_parameterisation.h"
#include <dx2/reflection.hpp>
#include "scan_static_predictor.cc"
#include "gradients_calculator.h"
#include <cmath>
#include <iostream>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;


class Target {
public:
    Target(
        Crystal &crystal,
        Goniometer &goniometer,
        MonochromaticBeam& beam,
        Panel panel,
        ReflectionTable& obs);
    std::vector<std::vector<double>> gradients() const;
    std::vector<double> residuals(std::vector<double> x);
    int nref() const;
    int nparams() const;
    SimpleBParameterisation B_parameterisation() const;
    SimpleUParameterisation U_parameterisation() const;
    SimpleDetectorParameterisation detector_parameterisation() const;
    SimpleBeamParameterisation beam_parameterisation() const;
    std::vector<double> rmsds() const;

private:
    Crystal crystal;
    Goniometer goniometer;
    MonochromaticBeam beam;
    Panel panel_;
    ReflectionTable& obs;
    SimpleBParameterisation Bparam;
    SimpleUParameterisation Uparam;
    SimpleDetectorParameterisation Dparam;
    SimpleBeamParameterisation Beamparam;
    GradientsCalculator calculator;
    int n_ref;
    int n_params;
    std::vector<double> rmsds_ = {0.0,0.0,0.0};
};

Target::Target(Crystal &crystal,
    Goniometer &goniometer,
    MonochromaticBeam& beam,
    Panel panel, ReflectionTable& obs):
crystal(crystal), goniometer(goniometer), beam(beam), panel_(panel), obs(obs),
Bparam(crystal), Uparam(crystal), Dparam(panel_), Beamparam(beam, goniometer), calculator(
    Uparam, Bparam, goniometer, Beamparam, Dparam){
        auto s1_ = obs.column<double>("s1");
        const auto& s1 = s1_.value();
        n_ref = s1.extent(0);
        n_params = Beamparam.get_params().size() + Uparam.get_params().size() + Bparam.get_params().size() + Dparam.get_params().size();
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

SimpleBParameterisation Target::B_parameterisation() const {
    return Bparam;
}

SimpleUParameterisation Target::U_parameterisation() const {
    return Uparam;
}

SimpleDetectorParameterisation Target::detector_parameterisation() const {
    return Dparam;
}

SimpleBeamParameterisation Target::beam_parameterisation() const {
    return Beamparam;
}

std::vector<double> Target::residuals(std::vector<double> x) {
    // Ok in future will need to split and set in the full set of parameterisations.
    std::vector<double> beam_params = {x[0], x[1], x[2]}; 
    std::vector<double> U_params = {x[3], x[4], x[5]}; 
    std::vector<double> B_params = {x[6], x[7], x[8], x[9], x[10], x[11]}; 
    std::vector<double> detector_params = {x[12], x[13], x[14], x[15], x[16], x[17]}; 
    Beamparam.set_params(beam_params);
    Uparam.set_params(U_params);
    Bparam.set_params(B_params);
    Dparam.set_params(detector_params);
    Matrix3d d = Dparam.get_state();
    panel_.update(d); //check maths here.
    Vector3d s0 = Beamparam.get_state();
    beam.set_s0(s0);
    Matrix3d B = Bparam.get_state();
    Matrix3d A = Uparam.get_state() * B;//crystal.get_B_matrix();
    simple_reflection_predictor(
        beam,
        goniometer,
        A,
        panel_,
        obs
    );
    auto xyzobs_ = obs.column<double>("xyzobs_mm");
    const auto& xyzobs_mm = xyzobs_.value();
    auto xyzcal_ = obs.column<double>("xyzcal_mm");
    const auto& xyzcal_mm = xyzcal_.value();
    std::vector<double> residuals(xyzobs_mm.size());// residuals vector is all dx then dy then dz
    int n = xyzobs_mm.extent(0);
    double xsum = 0;
    double ysum = 0;
    double zsum = 0;
    for (int i=0; i<n; ++i){
        residuals[i] = xyzcal_mm(i,0) - xyzobs_mm(i,0);
        xsum += std::pow(residuals[i], 2);
    }
    for (int i=n, k=0; i<(2*n); ++i, ++k){
        residuals[i] = xyzcal_mm(k,1) - xyzobs_mm(k,1);
        ysum += std::pow(residuals[i], 2);
    }
    for (int i=(2*n), k=0; i<(3*n); ++i, ++k){
        residuals[i] = xyzcal_mm(k,2) - xyzobs_mm(k,2);
        zsum += std::pow(residuals[i], 2);
    }
    rmsds_[0] = std::sqrt(xsum / n);
    rmsds_[1] = std::sqrt(ysum / n);
    rmsds_[2] = std::sqrt(zsum / n);
    //calculator = GradientsCalculator(crystal, goniometer, beam, panel, Dparam); // needed?
    return residuals;
}

std::vector<std::vector<double>> Target::gradients() const {
    return calculator.get_gradients(obs);
}