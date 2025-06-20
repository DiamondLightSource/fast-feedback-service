
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include "detector_parameterisation.h"
#include <dx2/reflection.hpp>
#include "scan_static_predictor.cc"
#include "gradients_calculator.h"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;


class Target {
public:
    Target(
        Crystal &crystal,
        Goniometer &goniometer,
        MonochromaticBeam& beam,
        Panel& panel,
        ReflectionTable& obs);
    std::vector<std::vector<double>> gradients() const;
    std::vector<double> residuals(std::vector<double> x);

private:
    Crystal crystal;
    Goniometer goniometer;
    MonochromaticBeam beam;
    Panel panel;
    ReflectionTable& obs;
    SimpleDetectorParameterisation Dparam;
    GradientsCalculator calculator;
};

Target::Target(Crystal &crystal,
    Goniometer &goniometer,
    MonochromaticBeam& beam,
    Panel& panel, ReflectionTable& obs):
crystal(crystal), goniometer(goniometer), beam(beam), panel(panel),
obs(obs), Dparam(SimpleDetectorParameterisation(panel)), calculator(
    crystal, goniometer, beam, panel, Dparam){};

std::vector<double> Target::residuals(std::vector<double> x) {
    // Ok in future will need to split and set in the full set of parameterisations.
    Dparam.set_params(x);
    Matrix3d d = Dparam.get_state();
    panel.update(d); //check maths here.
    simple_reflection_predictor(
        beam,
        goniometer,
        crystal.get_A_matrix(),
        panel,
        obs
    );
    auto xyzobs_ = obs.column<double>("xyzobs_mm");
    const auto& xyzobs_mm = xyzobs_.value();
    auto xyzcal_ = obs.column<double>("xyzcal_mm");
    const auto& xyzcal_mm = xyzcal_.value();
    std::vector<double> residuals(xyzobs_mm.size());// residuals vector is all dx then dy then dz
    int n = xyzobs_mm.extent(0);
    for (int i=0; i<n; ++i){
        residuals[i] = xyzcal_mm(i,0) - xyzobs_mm(i,0);
    }
    for (int i=n, k=0; i<(2*n); ++i, ++k){
        residuals[i] = xyzcal_mm(k,1) - xyzobs_mm(k,1);
    }
    for (int i=(2*n), k=0; i<(3*n); ++i, ++k){
        residuals[i] = xyzcal_mm(k,2) - xyzobs_mm(k,2);
    }
    calculator = GradientsCalculator(crystal, goniometer, beam, panel, Dparam); // needed?
    return residuals;
}

std::vector<std::vector<double>> Target::gradients() const {
    return calculator.get_gradients(obs);
}