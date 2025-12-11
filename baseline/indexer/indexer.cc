#include <assert.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/os.h>

#include <Eigen/Dense>
#include <algorithm>
#include <argparse/argparse.hpp>
#include <chrono>
#include <cstring>
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <dx2/scan.hpp>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "assign_indices.cc"
#include "combinations.cc"
#include "ffs_logger.hpp"
#include "fft3d.cc"
#include "flood_fill.cc"
#include "gemmi/symmetry.hpp"
#include "peaks_to_rlvs.cc"
#include "score_crystals.cc"
#include "xyz_to_rlp.cc"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;
using json = nlohmann::json;

constexpr double RAD2DEG = 180.0 / M_PI;
constexpr size_t indexed_flag = (1 << 2);  // 4
constexpr size_t strong_flag = (1 << 5);   // 32

int main(int argc, char **argv) {
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
    parser.add_argument("--no-output")
      .help("Suppress writing of reflection table")
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
        logger.error("Must specify experiment list file with --expt\n");
        std::exit(1);
    }
    if (!parser.is_used("refl")) {
        logger.error(
          "Must specify spotfinding results file (in DIALS HDF5 format) with --refl\n");
        std::exit(1);
    }
    // In DIALS, the max cell is automatically determined through a nearest
    // neighbour analysis that requires the annlib package. For now,
    // let's make this a required argument to help with testing/comparison
    // to DIALS.
    if (!parser.is_used("max-cell")) {
        logger.error("Must specify --max-cell\n");
        std::exit(1);
    }
    std::string imported_expt = parser.get<std::string>("expt");
    std::string filename = parser.get<std::string>("refl");
    double max_cell = parser.get<float>("max-cell");
    size_t max_refine = parser.get<size_t>("max-refine");

    // Parse the experiment list (a json file) and load the models.
    // Will be moved to dx2.
    std::ifstream f(imported_expt);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error &ex) {
        logger.error("Unable to read {}; json parse error at byte {}",
                     imported_expt.c_str(),
                     ex.byte);
        std::exit(1);
    }
    Experiment<MonochromaticBeam> expt;
    try {
        expt = Experiment<MonochromaticBeam>(elist_json_obj);
    } catch (std::invalid_argument const &ex) {
        logger.error("Unable to create MonochromaticBeam experiment: {}", ex.what());
        std::exit(1);
    }
    Scan scan = expt.scan();
    MonochromaticBeam beam = expt.beam();
    Goniometer gonio = expt.goniometer();
    Detector detector = expt.detector();
    assert(detector.panels().size()
           == 1);  // only considering single panel detectors initially.
    Panel panel = detector.panels()[0];

    // Read data from a reflection table.
    ReflectionTable strong_reflections(filename);
    auto xyzobs_px_ = strong_reflections.column<double>("xyzobs.px.value");
    if (!xyzobs_px_.has_value()) {
        throw std::runtime_error(
          "No xyzobs.px.value column found in input reflection table.");
    }
    mdspan_type<double> xyzobs_px = xyzobs_px_.value();
    if (xyzobs_px.rank() != 2 || xyzobs_px.extent(1) != 3) {
        throw std::runtime_error(
          "Input xyzobs.px.value column does not have the expected shape of (N,3).");
    }

    // The diffraction spots form a lattice in reciprocal space (if the experimental
    // geometry is accurate). So use the experimental models to transform the spot
    // coordinates on the detector into reciprocal space.
    xyz_to_rlp_results results = xyz_to_rlp(xyzobs_px, panel, beam, scan, gonio);
    logger.info("Number of reflections: {}", results.rlp.extent(0));

    // If a resolution limit was not specified, determine from the highest resolution spot.
    double d_min;
    if (parser.is_used("dmin")) {
        d_min = parser.get<float>("dmin");
    } else {
        std::vector<double> d_values(results.rlp.extent(0), 0);
        for (int i = 0; i < results.rlp.extent(0); ++i) {
            d_values[i] = 1.0 / Eigen::Map<Vector3d>(&results.rlp(i, 0)).norm();
        }
        d_min = *std::min_element(d_values.begin(), d_values.end());
        logger.info("Setting dmin based on highest resolution spot: {:.5f}", d_min);
    }

    // b_iso is an isotropic b-factor used to weight the points when doing the fft.
    // i.e. high resolution (weaker) spots are downweighted by the expected
    // intensity fall-off as as function of resolution.
    double b_iso = -4.0 * std::pow(d_min, 2) * log(0.05);
    uint32_t n_points = parser.get<uint32_t>("fft-npoints");
    logger.info("Setting b_iso = {:.3f}", b_iso);

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
      fft3d(results.rlp, real_fft_result, d_min, b_iso, n_points, nthreads);

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
        logger.info(
          "Insufficient number of candidate vectors to make a crystal model.");
        std::exit(0);
    }

    // Now we need to generate candidate crystals from the lattice vectors and evaluate them
    // FFS output does not yet have flags - so set all to strong.
    std::vector<std::size_t> flags(results.rlp.extent(0), strong_flag);
    // calculate entering array
    std::vector<bool> enterings(results.rlp.extent(0));
    Vector3d s0 = beam.get_s0();
    Vector3d axis = gonio.get_rotation_axis();
    Vector3d vec = s0.cross(axis);
    for (int i = 0; i < results.s1.extent(0); ++i) {
        Eigen::Map<Vector3d> s1_i(&results.s1(i, 0));
        enterings[i] = (s1_i.dot(vec)) < 0.0;
    }

    // Make a selection on dmin and rotation angle like dials
    std::vector<bool> selection(results.rlp.extent(0), true);
    double osc_trim_limit = scan.get_oscillation()[0] + 360.0;
    for (int i = 0; i < results.rlp.extent(0); ++i) {
        Eigen::Map<Vector3d> rlp_i(&results.rlp(i, 0));
        if (1.0 / rlp_i.norm() <= d_min) {
            selection[i] = false;
        } else if (results.xyzobs_mm(i, 2) * RAD2DEG > osc_trim_limit) {
            selection[i] = false;
        }
    }
    ReflectionTable reflections;
    reflections.add_column(std::string("flags"), flags.size(), 1, flags);
    reflections.add_column(std::string("xyzobs.mm.value"),
                           results.xyzobs_mm.extent(0),
                           3,
                           results.xyzobs_mm_data);
    reflections.add_column(std::string("s1"), results.s1.extent(0), 3, results.s1_data);
    reflections.add_column(
      std::string("rlp"), results.rlp.extent(0), 3, results.rlp_data);
    reflections.add_column(std::string("entering"), enterings);
    const ReflectionTable filtered = reflections.select(selection);

    Vector3i null{{0, 0, 0}};
    int n_images = scan.get_image_range()[1] - scan.get_image_range()[0] + 1;
    double scan_width =
      scan.get_oscillation()[0] + (scan.get_oscillation()[1] * n_images);

    // Initialise a class that will generate canditate orientation matrices.
    CandidateOrientationMatrices candidates(candidate_lattice_vectors, 1000);

    // Initialise a vector of threads
    std::vector<std::thread> threads;
    logger.info("Scoring {} candidate crystals using {} threads", max_refine, nthreads);

    // Limit the number of active threads to the max concurrency, without needing to manage a threadpool.
    int batch_size = std::min(max_refine, nthreads);
    int n = 0;  // Candidate number
    for (int i = 0; i < max_refine; i += nthreads) {
        threads.clear();
        int j = 0;
        while (candidates.has_next() && n < max_refine && j < batch_size) {
            std::optional<Crystal> next_crystal = candidates.next();
            if (next_crystal.has_value()) {
                Crystal crystal = next_crystal.value();
                n++;
                j++;
                threads.emplace_back(std::thread(evaluate_crystal,
                                                 crystal,
                                                 std::ref(filtered),
                                                 gonio,
                                                 beam,
                                                 panel,
                                                 scan_width,
                                                 n));
            } else {
                break;
            }
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    // Determine a relative score for all candidates and print a summary to the log.
    score_solutions(results_map);
    std::vector<std::pair<int, score_and_crystal>> results_vector(results_map.begin(),
                                                                  results_map.end());
    std::sort(
      results_vector.begin(), results_vector.end(), [](const auto &a, const auto &b) {
          return a.second.score < b.second.score;  // Ascending order by score
      });
    logger.info("Candidate solutions:");
    logger.info(
      "| Unit cell                                 | volume & score | #indexed % & "
      "score | rmsd_xy & score | overall score |");
    for (const auto &result : results_vector) {
        gemmi::UnitCell cell = result.second.crystal.get_unit_cell();
        logger.info(
          "| {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f} | {:>8.0f}  {:.2f} "
          "| {:>7.0f}  {:>3.0f}  {:.2f} | {:>6.2f}    {:>5.2f} |        {:>6.2f} |",
          cell.a,
          cell.b,
          cell.c,
          cell.alpha,
          cell.beta,
          cell.gamma,
          cell.volume,
          result.second.volume_score,
          result.second.num_indexed,
          result.second.fraction_indexed * 100,
          result.second.indexed_score,
          result.second.rmsdxy,
          result.second.rmsd_score,
          result.second.score);
    }
    Crystal best_xtal = results_vector[0].second.crystal;
    MonochromaticBeam best_beam = results_vector[0].second.beam;
    Panel best_panel = results_vector[0].second.panel;

    bool test = parser.get<bool>("test");
    if (test) {
        // dump the candidate vectors to json
        std::string n_vecs = std::to_string(candidate_lattice_vectors.size() - 1);
        size_t n_zero = n_vecs.length();
        json vecs_out;
        for (int i = 0; i < candidate_lattice_vectors.size(); ++i) {
            std::string s = std::to_string(i);
            auto pad_s = std::string(n_zero - std::min(n_zero, s.length()), '0') + s;
            vecs_out[pad_s] = candidate_lattice_vectors[i];
        }
        std::string outfile = "candidate_vectors.json";
        std::ofstream vecs_file(outfile);
        vecs_file << vecs_out.dump(4);
        logger.info("Saved candidate vectors to {}", outfile);

        size_t offset = std::to_string(results_vector.size() - 1).length();
        json crystals_out;
        for (int i = 0; i < results_vector.size(); ++i) {
            std::string s = std::to_string(i);
            auto pad_s = std::string(offset - std::min(offset, s.length()), '0') + s;
            crystals_out[pad_s] = results_vector[i].second.to_json();
        }
        std::string candidates_outfile = "candidate_crystals.json";
        std::ofstream candidates_file(candidates_outfile);
        candidates_file << crystals_out.dump(4);
        logger.info("Saved candidate crystals to {}", candidates_outfile);
    }

    // Update the experiment list models.
    expt.set_crystal(best_xtal);
    expt.beam().set_s0(best_beam.get_s0());
    expt.detector().update(best_panel.get_d_matrix());
    gemmi::UnitCell bestcell = best_xtal.get_unit_cell();
    logger.info("Best cell (P1): {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}",
                bestcell.a,
                bestcell.b,
                bestcell.c,
                bestcell.alpha,
                bestcell.beta,
                bestcell.gamma);

    // Save the indexed experiment list.
    json elist_out = expt.to_json();
    std::string efile_name = "indexed.expt";
    std::ofstream efile(efile_name);
    efile << elist_out.dump(4);
    logger.info("Saved experiment list to {}", efile_name);

    bool no_output = parser.get<bool>("no-output");
    if (!no_output) {
        // Load the strong refls, to persist existing data items
        std::vector<std::string> labels = {expt.identifier()};
        strong_reflections.set_identifiers(labels);

        // Recalculate the rlp and s1 vectors based on the updated models.
        xyz_to_rlp_results final_results =
          xyz_to_rlp(xyzobs_px, detector.panels()[0], expt.beam(), scan, gonio);
        strong_reflections.add_column(std::string("xyzobs.mm.value"),
                                      final_results.xyzobs_mm.extent(0),
                                      3,
                                      final_results.xyzobs_mm_data);
        strong_reflections.add_column(
          std::string("s1"), final_results.s1.extent(0), 3, final_results.s1_data);
        strong_reflections.add_column(
          std::string("rlp"), final_results.rlp.extent(0), 3, final_results.rlp_data);

        // Add required columns for next steps (DIALS compatibility)
        // if not panel, add
        auto panels = strong_reflections.column<uint64_t>("panel");
        if (!panels.has_value()) {
            std::vector<uint64_t> panel_column(final_results.rlp.extent(0), 0);
            strong_reflections.add_column(
              std::string("panel"), final_results.xyzobs_mm.extent(0), 1, panel_column);
        }
        auto xyzobs_px_var_ = strong_reflections.column<double>("xyzobs.px.variance");
        if (xyzobs_px_var_.has_value()) {
            // Need to convert the px variance to mm variance
            std::vector<double> xyzobs_mm_var_data(final_results.rlp.size());
            mdspan_type<double> xyzobs_mm_variance = mdspan_type<double>(
              xyzobs_mm_var_data.data(), xyzobs_mm_var_data.size() / 3, 3);
            auto xyzobs_px_var_ =
              strong_reflections.column<double>("xyzobs.px.variance");
            const auto &xyzobs_px_variance = xyzobs_px_var_.value();
            px_to_mm(xyzobs_px_variance, xyzobs_mm_variance, scan, best_panel);
            strong_reflections.add_column("xyzobs.mm.variance",
                                          xyzobs_mm_variance.extent(0),
                                          3,
                                          xyzobs_mm_var_data);
        }
        // Index the data with the refined models.
        assign_indices_results assign_results = assign_indices_global(
          expt.crystal().get_A_matrix(), final_results.rlp, final_results.xyzobs_mm);
        logger.info("Indexed {}/{} reflections using the refined models",
                    assign_results.number_indexed,
                    xyzobs_px.extent(0));
        strong_reflections.add_column(std::string("miller_index"),
                                      assign_results.miller_indices.extent(0),
                                      assign_results.miller_indices.extent(1),
                                      assign_results.miller_indices_data);
        // Set the indexed flag in the output.
        auto flags_ = strong_reflections.column<std::size_t>("flags");
        if (!flags_.has_value()) {
            for (int i = 0; i < assign_results.miller_indices.extent(0); ++i) {
                if ((assign_results.miller_indices(i, 0)) != 0
                    | (assign_results.miller_indices(i, 1) != 0)
                    | (assign_results.miller_indices(i, 2) != 0)) {
                    flags[i] |= indexed_flag;
                }
            }
            strong_reflections.add_column(std::string("flags"), flags);
        } else {
            auto flag_span = flags_.value();
            for (int i = 0; i < assign_results.miller_indices.extent(0); ++i) {
                if ((assign_results.miller_indices(i, 0)) != 0
                    | (assign_results.miller_indices(i, 1) != 0)
                    | (assign_results.miller_indices(i, 2) != 0)) {
                    flag_span(i, 0) |= indexed_flag;
                }
            }
        }

        // Save the indexed reflection table.
        std::string output_filename = "indexed.refl";
        strong_reflections.write(output_filename);
        logger.info("Saved reflection table to {}", output_filename);
    }

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    logger.info("Total time for indexer: {:.4f}s", elapsed_time.count());
}
