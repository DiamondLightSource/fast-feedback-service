#ifndef REFINE_CANDIDATE_H
#define REFINE_CANDIDATE_H
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>

#include "target.cc"
#include "detector_parameterisation.cc"
#include "U_parameterisation.cc"
#include "B_parameterisation.cc"

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

struct RefineFunctor
{
  Target& target;

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec)
  {
      // x has dimensions of the number of parameters (i.e. nx1)
      // fvec has dimensions of the number of observations i.e. mx1)
      // i.e. fvec is what DIALS would call the residuals vector.
      std::vector<double> x_vector(target.nparams());
      for (int i=0;i<target.nparams();++i){
        x_vector[i] = x(i);
      }
      std::vector<double> resids = target.residuals(x_vector); // This call does some recalculation.
      std::vector<double> rmsds = target.rmsds();
      double xyrmsd = std::sqrt(std::pow(rmsds[0], 2) + std::pow(rmsds[1],2));
      fvec = Eigen::Map<const Eigen::VectorXd>(resids.data(), resids.size());
      return 0;
  }

  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const {
    // x has dimensions of the number of parameters (i.e. nx1)
    // fjac has dimensions m x n i.e. the gradients vector with respect to
    // parameter i will be set in the column fjac(i,:).
    std::vector<std::vector<double>> gradients = target.gradients();
    for (int i=0;i<gradients.size();++i){
      fjac.col(i) = Eigen::Map<const Eigen::VectorXd>(gradients[i].data(), gradients[i].size());
    }
    return 0;
  }

  int inputs() const { return target.nparams(); } // inputs is the dimension of x.
  int values() const { return 3 * target.nref(); } // values is the number of residuals, (x,y,z for each refl here).
};

double refine_indexing_candidate(
  Crystal& crystal, const Goniometer& gonio, MonochromaticBeam& beam, Panel& panel, ReflectionTable& sel_obs
){
    Target target(crystal, gonio, beam, panel, sel_obs);
    Eigen::VectorXd x(target.nparams());
    SimpleBeamParameterisation beam_param = target.beam_parameterisation();
    std::vector<double> beamparams = beam_param.get_params();
    SimpleUParameterisation u_param = target.U_parameterisation();
    std::vector<double> uparams = u_param.get_params();
    SimpleBParameterisation B_param = target.B_parameterisation();
    std::vector<double> Bparams = B_param.get_params();
    SimpleDetectorParameterisation d_param = target.detector_parameterisation();
    std::vector<double> dparams = d_param.get_params();

    std::copy(beamparams.begin(), beamparams.end(), x.data());
    std::copy(uparams.begin(), uparams.end(), x.data()+3);
    std::copy(Bparams.begin(), Bparams.end(), x.data()+6);
    std::copy(dparams.begin(), dparams.end(), x.data()+12);

    RefineFunctor minimiser(target);
    Eigen::LevenbergMarquardt<RefineFunctor, double> levenbergMarquardt(minimiser);

    levenbergMarquardt.parameters.ftol = 1e-6;
    levenbergMarquardt.parameters.xtol = 1e-6;
    levenbergMarquardt.parameters.maxfev = 10; // Max iterations

    Eigen::VectorXd xmin = x; // initialize
    levenbergMarquardt.minimize(xmin);

    std::vector<double> rmsds = target.rmsds();
    double xyrmsd = std::sqrt(std::pow(rmsds[0], 2) + std::pow(rmsds[1],2));
    // Update the crystal model. The beam and detector models have already been
    // updated during the refinement
    crystal.set_A_matrix(
      target.U_parameterisation().get_state() *
      target.B_parameterisation().get_state()); 
    return xyrmsd;
}

#endif
