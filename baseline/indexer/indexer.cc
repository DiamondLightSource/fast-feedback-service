#include <assert.h>
#include <dx2/beam.h>
#include <dx2/crystal.h>
#include <dx2/detector.h>
#include <dx2/experiment.h>
#include <dx2/goniometer.h>
#include <dx2/scan.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/os.h>
#include <mutex>

#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <chrono>
#include <cstring>
#include <dx2/h5/h5read_processed.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"
#include "fft3d.cc"
#include "flood_fill.cc"
#include "gemmi/symmetry.hpp"
#include "peaks_to_rlvs.cc"
#include "xyz_to_rlp.cc"

#include "refman_filter.cc"
#include "assign_indices.h"
#include "reflection_data.h"
#include "scanstaticpredictor.cc"
#include "combinations.cc"
#include "non_primitive_basis.cc"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;
using json = nlohmann::json;

std::mutex mtx;

struct score_and_crystal {
    double score;
    Crystal crystal;
    double num_indexed;
    double rmsdxy;
    double fraction_indexed;
    double volume_score;
    double indexed_score;
    double rmsd_score;

    json to_json(){
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

std::map<int,score_and_crystal> results_map;
    
void calc_score(Crystal crystal,
  reflection_data const& obs,
  Goniometer gonio, MonochromaticBeam beam, Panel panel, double width, int n){
  std::vector<Vector3i> miller_indices;
  int count;
  auto preassign = std::chrono::system_clock::now();
  std::tie(miller_indices, count) = assign_indices_global(crystal.get_A_matrix(), obs.rlp, obs.xyzobs_mm);
  auto t2 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = t2 - preassign;
  logger->debug("Time for assigning indices: {:.5f}s", elapsed_time.count());
  //obs.miller_indices = miller_indices;
  auto t3 = std::chrono::system_clock::now();
  // Perform the non-primivite basis correction.
  count = correct(miller_indices, crystal, obs.rlp, obs.xyzobs_mm);
  auto t4 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time1 = t4 - t3;
  logger->debug("Time for correct: {:.5f}s", elapsed_time1.count());
  auto prefilter = std::chrono::system_clock::now();
  reflection_data sel_obs = reflection_filter_preevaluation(
      obs, miller_indices, gonio, crystal, beam, panel, width, 20
  );
  auto postfilter = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_timefilter = postfilter - prefilter;
  logger->debug("Time for reflection_filter: {:.5f}s", elapsed_timefilter.count());
  
  //write the score to the results map
  double xsum = 0;
  double ysum = 0;
  double zsum = 0;
  for (int i=0;i<sel_obs.flags.size();i++){
      Vector3d xyzobs = sel_obs.xyzobs_mm[i];
      Vector3d xyzcal = sel_obs.xyzcal_mm[i];
      xsum += std::pow(xyzobs[0] - xyzcal[0],2);
      ysum += std::pow(xyzobs[1] - xyzcal[1],2);
      zsum += std::pow(xyzobs[2] - xyzcal[2],2);
  }
  double rmsdx = std::sqrt(xsum / sel_obs.xyzcal_mm.size());
  double rmsdy = std::sqrt(ysum / sel_obs.xyzcal_mm.size());
  double rmsdz = std::sqrt(zsum / sel_obs.xyzcal_mm.size());
  double xyrmsd = std::sqrt(rmsdx * rmsdx + rmsdy * rmsdy);

  score_and_crystal sac;
  sac.score = n;
  sac.crystal = crystal;
  sac.num_indexed = count;
  sac.rmsdxy = xyrmsd;
  sac.fraction_indexed = (double)count / obs.flags.size();
  logger->info("Scored candidate crystal {}", n);
  mtx.lock();
  results_map[n] = sac;
  mtx.unlock();
}

void score_solutions(std::map<int,score_and_crystal>& results_map){
  // Score the refined models.
  // Score is defined as volume_score + rmsd_score + fraction_indexed_score
  int length = results_map.size();
  std::vector<double> rmsd_scores(length, 0);
  std::vector<double> volume_scores(length, 0);
  std::vector<double> fraction_indexed_scores(length, 0);
  std::vector<double> combined_scores(length, 0);
  double log2 = std::log(2);
  for (int i=0;i<length;++i){
    rmsd_scores[i] = std::log(results_map[i+1].rmsdxy) / log2;
    fraction_indexed_scores[i] = std::log(results_map[i+1].fraction_indexed) / log2;
    volume_scores[i] = std::log(results_map[i+1].crystal.get_unit_cell().volume) / log2;
  }
  double min_rmsd_score = *std::min_element(rmsd_scores.begin(), rmsd_scores.end());
  double max_frac_score = *std::max_element(fraction_indexed_scores.begin(), fraction_indexed_scores.end());
  double min_volume_score = *std::min_element(volume_scores.begin(), volume_scores.end());
  for (int i=0;i<length;++i){
    rmsd_scores[i] -= min_rmsd_score;
    results_map[i+1].rmsd_score = rmsd_scores[i];
    fraction_indexed_scores[i] *= -1;
    fraction_indexed_scores[i] += max_frac_score;
    results_map[i+1].indexed_score = fraction_indexed_scores[i];
    volume_scores[i] -= min_volume_score;
    results_map[i+1].volume_score = volume_scores[i];
  }
  for (int i=0;i<length;++i){
    results_map[i+1].score = rmsd_scores[i] + fraction_indexed_scores[i] + volume_scores[i];
  }
}



constexpr double RAD2DEG = 180.0 / M_PI;

int main(int argc, char** argv) {
    // The purpose of an indexer is to determine the lattice model that best
    // explains the positions of the strong spots found during spot-finding.
    // The lattice model is a set of three vectors that define the crystal
    // lattice translations.
    // The experiment models (beam, detector) can also be refined during the
    // indexing process. The output is a set of models - a new crystal model that
    // describes the crystal lattice and an updated set of experiment models.
    auto t1 = std::chrono::system_clock::now();
    auto parser = argparse::ArgumentParser();
    parser.add_argument("-e", "--expt").help("Path to the DIALS expt file");
    parser.add_argument("-r", "--refl")
      .help("Path to the h5 reflection table file containing spotfinding results");
    parser.add_argument("--dmin")
      .help("The resolution limit of spots to use in the indexing process.")
      .scan<'f', float>();
    parser.add_argument("--max-cell")
      .help("The maximum possible cell length to consider during indexing")
      .scan<'f', float>();
    parser.add_argument("--max-refine")
      .help("The maximum number of candidate lattices to refine during indexing")
      .default_value<size_t>(50)
      .scan<'u', size_t>();
    parser.add_argument("--test")
      .help("Enable additional output for testing")
      .default_value<bool>(false)
      .implicit_value(true);
    parser
      .add_argument(
        "--fft-npoints")  // mainly for testing, likely would always want to keep it as 256.
      .help(
        "The number of grid points to use for the fft. Powers of two are most "
        "efficient.")
      .default_value<uint32_t>(256)
      .scan<'u', uint32_t>();
    parser
      .add_argument("--nthreads")  // mainly for testing.
      .help(
        "The number of threads to use for the fft calculation."
        "Defaults to the value of std::thread::hardware_concurrency."
        "Better performance can typically be obtained with a higher number"
        "of threads than this.")
      .scan<'u', size_t>();
    parser.parse_args(argc, argv);

    if (!parser.is_used("expt")) {
        logger->error("Must specify experiment list file with --expt\n");
        std::exit(1);
    }
    if (!parser.is_used("refl")) {
        logger->error(
          "Must specify spotfinding results file (in DIALS HDF5 format) with --refl\n");
        std::exit(1);
    }
    // In DIALS, the max cell is automatically determined through a nearest
    // neighbour analysis that requires the annlib package. For now,
    // let's make this a required argument to help with testing/comparison
    // to DIALS.
    if (!parser.is_used("max-cell")) {
        logger->error("Must specify --max-cell\n");
        std::exit(1);
    }
    // FIXME use highest resolution by default to remove this requirement.
    if (!parser.is_used("dmin")) {
        logger->error("Must specify --dmin\n");
        std::exit(1);
    }
    std::string imported_expt = parser.get<std::string>("expt");
    std::string filename = parser.get<std::string>("refl");
    double max_cell = parser.get<float>("max-cell");
    double d_min = parser.get<float>("dmin");
    size_t max_refine = parser.get<size_t>("max-refine");

    // Parse the experiment list (a json file) and load the models.
    // Will be moved to dx2.
    std::ifstream f(imported_expt);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error& ex) {
        logger->error("Unable to read {}; json parse error at byte {}",
                      imported_expt.c_str(),
                      ex.byte);
        std::exit(1);
    }
    Experiment<MonochromaticBeam> expt;
    try {
        expt = Experiment<MonochromaticBeam>(elist_json_obj);
    } catch (std::invalid_argument const& ex) {
        logger->error("Unable to create MonochromaticBeam experiment: {}", ex.what());
        std::exit(1);
    }
    Scan scan = expt.scan();
    MonochromaticBeam beam = expt.beam();
    Goniometer gonio = expt.goniometer();
    Detector detector = expt.detector();
    assert(detector.panels().size()
           == 1);  // only considering single panel detectors initially.
    Panel panel = detector.panels()[0];

    // Read data from a reflection table. Again, this should be moved to
    // dx2 and only require the data array name (xyzobs.px.value) with some
    // logic to step through the directory structure
    std::string array_name = "/dials/processing/group_0/xyzobs.px.value";
    // Note, xyzobs_px is the flattened, on-disk representation of the array
    // i.e. if there are 100 spots, the length of xyzobs_px is 300, and
    // contains the elements [x0, y0, z0, x1, y1, z1, ..., x99, y99, z99]
    std::vector<double> xyzobs_px =
      read_array_from_h5_file<double>(filename, array_name);

    // The diffraction spots form a lattice in reciprocal space (if the experimental
    // geometry is accurate). So use the experimental models to transform the spot
    // coordinates on the detector into reciprocal space.
    std::vector<Vector3d> rlp;
    std::vector<Vector3d> s1;
    std::vector<Vector3d> xyzobs_mm;
    
    std::tie(rlp, s1, xyzobs_mm)= xyz_to_rlp(xyzobs_px, panel, beam, scan, gonio);
    logger->info("Number of reflections: {}", rlp.size());

    // b_iso is an isotropic b-factor used to weight the points when doing the fft.
    // i.e. high resolution (weaker) spots are downweighted by the expected
    // intensity fall-off as as function of resolution.
    double b_iso = -4.0 * std::pow(d_min, 2) * log(0.05);
    uint32_t n_points = parser.get<uint32_t>("fft-npoints");
    logger->info("Setting b_iso = {:.3f}", b_iso);

    // Create an array to store the fft result. This is a 3D grid of points, typically 256^3.
    std::vector<double> real_fft_result(n_points * n_points * n_points);

    // Do the fft of the reciprocal lattice coordinates.
    // the used in indexing array denotes whether a coordinate was used for the
    // fft (might not be if dmin filter was used for example). The used_in_indexing array
    // is sometimes used onwards in the dials indexing algorithms, so keep for now.
    size_t nthreads;
    if (parser.is_used("nthreads")) {
        nthreads = parser.get<size_t>("nthreads");
    } else {
        size_t max_threads = std::thread::hardware_concurrency();
        nthreads = max_threads ? max_threads : 1;
    }

    std::vector<bool> used_in_indexing =
      fft3d(rlp, real_fft_result, d_min, b_iso, n_points, nthreads);

    // The fft result is noisy. We want to extract the peaks, which may be spread over several
    // points on the fft grid. So we use a flood fill algorithm (https://en.wikipedia.org/wiki/Flood_fill)
    // to determine the connected regions in 3D. This is how it is done in DIALS, but I note that
    // perhaps this could be done with connected components analysis.
    // So do the flood fill, and extract the centres of mass of the peaks and the number of grid points
    // that contribute to each peak.
    std::vector<int> grid_points_per_peak;
    std::vector<Vector3d> fractional_centres_of_mass;
    // 15.0 is the DIALS 'rmsd_cutoff' parameter to filter out weak peaks.
    std::tie(grid_points_per_peak, fractional_centres_of_mass) =
      flood_fill(real_fft_result, 15.0, n_points);
    // Do some further filtering, 0.15 is the DIALS peak_volume_cutoff parameter.
    std::tie(grid_points_per_peak, fractional_centres_of_mass) =
      flood_fill_filter(grid_points_per_peak, fractional_centres_of_mass, 0.15);

    // Convert the peak centres from the fft grid into vectors in reciprocal space. These are our candidate
    // lattice vectors.
    // 3.0 is the min cell parameter.
    std::vector<Vector3d> candidate_lattice_vectors = peaks_to_rlvs(
      fractional_centres_of_mass, grid_points_per_peak, d_min, 3.0, max_cell, n_points);

    if (candidate_lattice_vectors.size() < 3) {
      logger->info("Insufficient number of candidate vectors to make a crystal model.");
      std::exit(0);
    }

    // at this point, we will test combinations of the candidate vectors, use those to index the spots, do some
    // refinement of the candidates and choose the best one. Then we will do some more refinement including extra
    // model parameters. At then end, we will have a list of refined experiment models (including a crystal)

    // For now, let's just write out the candidate vectors and write out the unrefined experiment models with the
    // first combination of candidate vectors as an example crystal, to demonstrate an example experiment list data
    // structure.

    std::string flags_array_name = "/dials/processing/group_0/flags";
    std::vector<std::size_t> flags = read_array_from_h5_file<std::size_t>(filename, flags_array_name);
    // calculate entering array
    std::vector<bool> enterings(rlp.size());
    Vector3d s0 = beam.get_s0();
    Vector3d axis = gonio.get_rotation_axis();
    Vector3d vec = s0.cross(axis);
    for (int i=0;i<s1.size();i++){
        enterings[i] = ((s1[i].dot(vec)) < 0.0);
    }

    // Make a selection on dmin and rotation angle like dials
    std::vector<bool> selection(rlp.size(), true);
    for (int i=0;i<rlp.size();i++){
      if (1.0/rlp[i].norm() <= d_min){
        selection[i] = false;
      }
      else if (xyzobs_mm[i][2]*RAD2DEG > 360.0){
        selection[i] = false;
      }
    }
    reflection_data reflections;
    reflections.flags = flags;
    reflections.xyzobs_mm = xyzobs_mm;
    reflections.s1 = s1;
    reflections.entering=enterings;
    reflections.rlp = rlp;
    reflections = select(reflections, selection);

    Vector3i null{{0,0,0}};
    int n = 0;
    // FIXME check somewhere that there are solutions
    CandidateOrientationMatrices candidates(candidate_lattice_vectors, 1000);
    // iterate over candidates; assign indices, refine, score.
    // need a map of scores for candidates: index to score and xtal. What about miller indices?
    std::vector<Vector3i> miller_indices;
    int count;
    int n_images = scan.get_image_range()[1] - scan.get_image_range()[0] + 1;
    double width = scan.get_oscillation()[0] + (scan.get_oscillation()[1] * n_images);

    std::vector<std::thread> threads;
    // Limit the number of active threads to the max concurrency, without needing to manage a threadpool.
    int batch_size = std::min(max_refine, nthreads);
    for (int i=0; i<max_refine; i += nthreads){
      threads.clear();
      int j=0;
      while (candidates.has_next() && n < max_refine && j < batch_size){
        Crystal crystal = candidates.next(); //quick (<0.1ms)
        n++;
        j++;
        threads.emplace_back(std::thread(calc_score, crystal, reflections, gonio, beam, panel, width, n));
      }
      for (auto &t : threads){
          t.join();
      }
    }

    score_solutions(results_map);
    std::vector<std::pair<int, score_and_crystal>> results_vector(results_map.begin(), results_map.end());
    std::sort(results_vector.begin(), results_vector.end(), [](const auto& a, const auto& b) {
        return a.second.score < b.second.score; // Ascending order by score
    });
    logger->info("Candidate solutions:");
    logger->info("| Unit cell                                 | volume & score | #indexed % & score | rmsd_xy & score | overall score |");
    for (const auto& result: results_vector){
        gemmi::UnitCell cell = result.second.crystal.get_unit_cell();
        logger->info("| {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f} | {:>8.0f}  {:.2f} | {:>7.0f}  {:>3.0f}  {:.2f} | {:>6.2f}    {:>5.2f} |        {:>6.2f} |",
          cell.a,cell.b,cell.c,cell.alpha,cell.beta,cell.gamma,
          cell.volume, result.second.volume_score,
          result.second.num_indexed,
          result.second.fraction_indexed*100,
          result.second.indexed_score,
          result.second.rmsdxy,
          result.second.rmsd_score,
          result.second.score);
    }
    Crystal best_xtal = results_vector[0].second.crystal;

    bool test = parser.get<bool>("test");
    if (test){
      // dump the candidate vectors to json
      std::string n_vecs = std::to_string(candidate_lattice_vectors.size() - 1);
      size_t n_zero = n_vecs.length();
      json vecs_out;
      for (int i = 0; i < candidate_lattice_vectors.size(); i++) {
          std::string s = std::to_string(i);
          auto pad_s = std::string(n_zero - std::min(n_zero, s.length()), '0') + s;
          vecs_out[pad_s] = candidate_lattice_vectors[i];
      }
      std::string outfile = "candidate_vectors.json";
      std::ofstream vecs_file(outfile);
      vecs_file << vecs_out.dump(4);
      logger->info("Saved candidate vectors to {}", outfile);

      size_t offset = std::to_string(results_vector.size() - 1).length();
      json crystals_out;
      for (int i = 0; i < results_vector.size(); i++) {
        std::string s = std::to_string(i);
        auto pad_s = std::string(offset - std::min(offset, s.length()), '0') + s;
        crystals_out[pad_s] = results_vector[i].second.to_json();
      }
      std::string candidates_outfile = "candidate_crystals.json";
      std::ofstream candidates_file(candidates_outfile);
      candidates_file << crystals_out.dump(4);
      logger->info("Saved candidate crystals to {}", candidates_outfile);
    }

    // Now save an experiment list with the models.
    expt.set_crystal(best_xtal);
    json elist_out = expt.to_json();
    std::string efile_name = "elist.json";
    std::ofstream efile(efile_name);
    efile << elist_out.dump(4);
    logger->info("Saved experiment list to {}", efile_name);

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    logger->info("Total time for indexer: {:.4f}s", elapsed_time.count());
}
