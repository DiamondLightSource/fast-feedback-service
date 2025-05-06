#include <dx2/beam.h>
#include <dx2/crystal.h>
#include <dx2/detector.h>
#include <dx2/experiment.h>
#include <dx2/goniometer.h>

#include <chrono>
#include <mutex>
#include <nlohmann/json.hpp>
#include <vector>

#include "assign_indices.cc"
#include "non_primitive_basis.cc"
#include "reflection_data.cc"
#include "reflection_filter.cc"

std::mutex score_and_crystal_mtx;

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
                      reflection_data const& obs,
                      Goniometer gonio,
                      MonochromaticBeam beam,
                      Panel panel,
                      double scan_width,
                      int n) {
    std::vector<Vector3i> miller_indices;
    int count;
    auto preassign = std::chrono::system_clock::now();

    // First assign miller indices to the data using the crystal model.
    std::tie(miller_indices, count) =
      assign_indices_global(crystal.get_A_matrix(), obs.rlp, obs.xyzobs_mm);
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - preassign;
    logger->debug("Time for assigning indices: {:.5f}s", elapsed_time.count());

    // Perform the (potential) non-primivite basis correction.
    count = correct(miller_indices, crystal, obs.rlp, obs.xyzobs_mm);
    auto t3 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time1 = t3 - t2;
    logger->debug("Time for correct: {:.5f}s", elapsed_time1.count());

    // Perform filtering of the data prior to candidate refinement.
    reflection_data sel_obs = reflection_filter_preevaluation(
      obs, miller_indices, gonio, crystal, beam, panel, scan_width, 20);
    auto postfilter = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_timefilter = postfilter - t3;
    logger->debug("Time for reflection_filter: {:.5f}s", elapsed_timefilter.count());

    // Refinement will go here in future.

    // Calculate rmsds
    double xsum = 0;
    double ysum = 0;
    double zsum = 0;
    for (int i = 0; i < sel_obs.flags.size(); i++) {
        Vector3d xyzobs = sel_obs.xyzobs_mm[i];
        Vector3d xyzcal = sel_obs.xyzcal_mm[i];
        xsum += std::pow(xyzobs[0] - xyzcal[0], 2);
        ysum += std::pow(xyzobs[1] - xyzcal[1], 2);
        zsum += std::pow(xyzobs[2] - xyzcal[2], 2);
    }
    double rmsdx = std::sqrt(xsum / sel_obs.xyzcal_mm.size());
    double rmsdy = std::sqrt(ysum / sel_obs.xyzcal_mm.size());
    double rmsdz = std::sqrt(zsum / sel_obs.xyzcal_mm.size());
    double xyrmsd = std::sqrt(rmsdx * rmsdx + rmsdy * rmsdy);

    // Write some data to the results map
    score_and_crystal sac;
    sac.crystal = crystal;
    sac.num_indexed = count;
    sac.rmsdxy = xyrmsd;
    sac.fraction_indexed = (double)count / obs.flags.size();
    logger->info("Scored candidate crystal {}", n);
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
