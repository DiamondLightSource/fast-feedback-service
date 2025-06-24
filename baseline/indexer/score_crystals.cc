#ifndef SCORE_CRYSTALS_H
#define SCORE_CRYSTALS_H

#include <chrono>
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <mutex>
#include <nlohmann/json.hpp>
#include <vector>
#include <algorithm>
#include "assign_indices.cc"
#include "ffs_logger.hpp"
#include "non_primitive_basis.cc"
#include "reflection_filter.cc"
#include "target.h"
#include "detector_parameterisation.h"

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

std::mutex score_and_crystal_mtx;

// Define target function
// Needs to get the model state, use it to update detector model
// then call the simple_predictor and calculate residuals
// also calculate gradients and assign into a jacobian matrix.

struct RefineFunctor
{
  Target& target;
  typedef float Scalar;

  typedef Eigen::VectorXd InputType;
  typedef Eigen::VectorXd ValueType;
  typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> JacobianType;

  enum {
      InputsAtCompileTime = Eigen::Dynamic,
      ValuesAtCompileTime = Eigen::Dynamic
  };

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec)
  {
      // x has dimensions of the number of parameters (i.e. nx1)
      // fvec has dimensions of the number of observations i.e. mx1)
      // i.e. fvec is what we would call the residuals vector.
      // target.residuals(x) - will know how to set params values, repredict, then
      // calculate xyz residuals
      std::vector<double> x_vector(target.nparams());
      for (int i=0;i<target.nparams();++i){
        x_vector[i] = x(i);
      }
      std::vector<double> resids = target.residuals(x_vector);
      std::vector<double> rmsds = target.rmsds();
      double xyrmsd = std::sqrt(std::pow(rmsds[0], 2) + std::pow(rmsds[1],2));
      logger.info("RMSDXY {:.5f} ", xyrmsd);
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

  int inputs() const { return target.nparams(); }// inputs is the dimension of x.
  int values() const { return 3 * target.nref(); } // "values" is the number of f_i and
};


// A struct to score a candidate crystal model.
struct score_and_crystal {
    double score;
    Crystal crystal;
    double num_indexed;
    double rmsdxy;
    double fraction_indexed;
    double volume_score;
    double indexed_score;
    double rmsd_score;

    json to_json() {
        json data;
        data["score"] = score;
        data["num_indexed"] = num_indexed;
        data["rmsdxy"] = rmsdxy;
        data["fraction_indexed"] = fraction_indexed;
        data["volume_score"] = volume_score;
        data["indexed_score"] = indexed_score;
        data["rmsd_score"] = rmsd_score;
        data["crystal"] = crystal.to_json();
        return data;
    }
};

std::map<int, score_and_crystal> results_map;

/**
 * @brief Evaluate a crystal model by evaluating how well it describes the reflection data.
 * @param crystal The crystal model.
 * @param obs The reflection data.
 * @param gonio The goniometer model.
 * @param beam The beam model.
 * @param panel The panel from the detector model.
 * @param scan_width The scan width in degrees.
 * @param n The candidate number, starting at 1.
 */
void evaluate_crystal(Crystal crystal,
                      ReflectionTable const& obs,
                      Goniometer gonio,
                      MonochromaticBeam beam,
                      Panel panel,
                      double scan_width,
                      int n) {
    std::vector<int> miller_indices_data;
    int count;
    auto preassign = std::chrono::system_clock::now();

    // First assign miller indices to the data using the crystal model.
    auto rlp_ = obs.column<double>("rlp");
    const mdspan_type<double>& rlp = rlp_.value();
    auto xyzobs_mm_ = obs.column<double>("xyzobs_mm");
    const mdspan_type<double>& xyzobs_mm = xyzobs_mm_.value();
    assign_indices_results results =
      assign_indices_global(crystal.get_A_matrix(), rlp, xyzobs_mm);
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - preassign;
    logger.debug("Time for assigning indices: {:.5f}s", elapsed_time.count());

    // Perform the (potential) non-primivite basis correction.
    count = correct(results.miller_indices_data, crystal, rlp, xyzobs_mm);

    auto t3 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time1 = t3 - t2;
    logger.debug("Time for correct: {:.5f}s", elapsed_time1.count());

    // Perform filtering of the data prior to candidate refinement.
    ReflectionTable sel_obs = reflection_filter_preevaluation(
      obs, results.miller_indices, gonio, crystal, beam, panel, scan_width, 20);
    auto postfilter = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_timefilter = postfilter - t3;
    logger.debug("Time for reflection_filter: {:.5f}s", elapsed_timefilter.count());

    // Refinement will go here in future.
    // First make CrystalOrientationParameterisation, CrystalUnitCellParameterisation, BeamParameterisation,
    // DetectorParameterisationSinglePanel,
    // Then make SimplePredictionParam
    // Then set the crystal U, B in the expt object.
    // gradients = pred.get_gradients(obs)
    
    // Encapsulate those in a simple target, with a parameter vector.
    // That can calc resids and gradients which are input to a least-squares routine - use Eigen (lev mar)?
    // As part of residuals, it updates the parameterisation objects, updates the experiment objects
    // and runs the simple predictor on the reflection data.
    
    Target target(crystal, gonio, beam, panel, sel_obs);
    Eigen::VectorXd x(9);
    SimpleBeamParameterisation beam_param = target.beam_parameterisation();
    std::vector<double> beamparams = beam_param.get_params();
    SimpleDetectorParameterisation d_param = target.detector_parameterisation();
    std::vector<double> dparams = d_param.get_params();
    x(0) = beamparams[0];
    x(1) = beamparams[1];
    x(2) = beamparams[2];
    x(3) = dparams[0];
    x(4) = dparams[1];
    x(5) = dparams[2];
    x(6) = dparams[3];
    x(7) = dparams[4];
    x(8) = dparams[5];
    logger.info("Initial beam params: {:.4f}, {:.4f}, {:.4f}", x(0), x(1), x(2));

    logger.info("Initial detector params: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}", x(3), x(4), x(5), x(6), x(7), x(8));

    RefineFunctor minimiser(target);
    Eigen::LevenbergMarquardt<RefineFunctor, double> levenbergMarquardt(minimiser);

    levenbergMarquardt.parameters.ftol = 1e-6;
    levenbergMarquardt.parameters.xtol = 1e-6;
    levenbergMarquardt.parameters.maxfev = 10; // Max iterations

    Eigen::VectorXd xmin = x; // initialize
    levenbergMarquardt.minimize(xmin);
    logger.info("Minimsed beam params: {:.4f}, {:.4f}, {:.4f}", xmin(0), xmin(1), xmin(2));

    logger.info("Minimsed detector params: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}", xmin(3), xmin(4), xmin(5), xmin(6), xmin(7), xmin(8));

    std::vector<double> rmsds = target.rmsds();
    double xyrmsd = std::sqrt(std::pow(rmsds[0], 2) + std::pow(rmsds[1],2));

    // Write some data to the results map
    score_and_crystal sac;
    sac.crystal = crystal;
    sac.num_indexed = count;
    sac.rmsdxy = xyrmsd;
    sac.fraction_indexed = static_cast<double>(count) / rlp.extent(0);
    logger.info("Scored candidate crystal {}", n);
    score_and_crystal_mtx.lock();
    results_map[n] = sac;
    score_and_crystal_mtx.unlock();
}

/**
 * @brief Determine a relative score for all solutions.
 * @param results_map A map of the candidate number to its score_and_crystal struct.
 */
void score_solutions(std::map<int, score_and_crystal>& results_map) {
    // Score the refined models.
    // Score is defined as volume_score + rmsd_score + fraction_indexed_score
    int length = results_map.size();
    std::vector<double> rmsd_scores(length, 0);
    std::vector<double> volume_scores(length, 0);
    std::vector<double> fraction_indexed_scores(length, 0);
    std::vector<double> combined_scores(length, 0);
    double log2 = std::log(2);
    for (int i = 0; i < length; ++i) {
        rmsd_scores[i] = std::log(results_map[i + 1].rmsdxy) / log2;
        fraction_indexed_scores[i] =
          std::log(results_map[i + 1].fraction_indexed) / log2;
        volume_scores[i] =
          std::log(results_map[i + 1].crystal.get_unit_cell().volume) / log2;
    }
    double min_rmsd_score = *std::min_element(rmsd_scores.begin(), rmsd_scores.end());
    double max_frac_score =
      *std::max_element(fraction_indexed_scores.begin(), fraction_indexed_scores.end());
    double min_volume_score =
      *std::min_element(volume_scores.begin(), volume_scores.end());
    for (int i = 0; i < length; ++i) {
        rmsd_scores[i] -= min_rmsd_score;
        results_map[i + 1].rmsd_score = rmsd_scores[i];
        fraction_indexed_scores[i] *= -1;
        fraction_indexed_scores[i] += max_frac_score;
        results_map[i + 1].indexed_score = fraction_indexed_scores[i];
        volume_scores[i] -= min_volume_score;
        results_map[i + 1].volume_score = volume_scores[i];
    }
    for (int i = 0; i < length; ++i) {
        results_map[i + 1].score =
          rmsd_scores[i] + fraction_indexed_scores[i] + volume_scores[i];
    }
}

#endif
