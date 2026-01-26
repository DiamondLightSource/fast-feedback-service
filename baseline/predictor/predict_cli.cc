/**
 * @file predict.cc
 * @brief This program implements the algorithm for spot prediction.
 *
 * ReekeIndexGenerator outlined in in the LURE workshop notes.
 *
 */

#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <chrono>
#include <cmath>
#include <common.hpp>
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <dx2/scan.hpp>
#include <exception>
#include <fstream>
#include <gemmi/symmetry.hpp>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include "index_generators.cc"
#include "predict.cc"
#include "ray_predictors.cc"
#include "threadpool.cc"
#include "utils.cc"

using json = nlohmann::json;

using Eigen::Matrix3d;
using Eigen::Vector3d;

#pragma region Argument Parser Configuration
/**
 * @brief Take a default-initialized ArgumentParser object and configure it
 *      with the arguments to be parsed; assign various properties to each
 *      argument, eg. help message, default value, etc.
 *
 * @param parser The ArgumentParser object (pre-input) to be configured.
 */
void configure_parser(argparse::ArgumentParser &parser) {
    parser.add_argument("-e", "--expt").help("path to DIALS expt file");
    parser.add_argument("--dmin")
      .help("minimum d-spacing of predicted reflections")
      .scan<'f', double>()
      .default_value(-1.0);
    parser.add_argument("-b", "--buffer_size")
      .help(
        "calculates predictions within a buffer zone of n images either side"
        "of the scan (for static)")
      .scan<'i', int>()
      .default_value<int>(0);
    parser.add_argument("-s", "--force_static")
      .help("for a scan varying model, forces static prediction")
      .default_value(false)
      .implicit_value(true);
    parser.add_argument("-n", "--nthreads")
      .help("Number of threads for parallelisation")
      .scan<'u', size_t>();
}

/**
 * @brief Take an ArgumentParser object after the user has entered input and check
 *      it for consistency; output errors and exit the program if a check fails.
 *
 * @param parser The ArgumentParser object (post-input) to be verified.
 */
void verify_arguments(const argparse::ArgumentParser &parser) {
    if (!parser.is_used("expt")) {
        logger.error("Must specify experiment list file with -e or --expt\n");
        std::exit(1);
    }
    if (parser.is_used("buffer_size") && parser.get<int>("buffer_size") < 0) {
        logger.error("--buffer_size cannot be negative\n");
    }
}
#pragma endregion

int main(int argc, char **argv) {
    auto t1 = std::chrono::system_clock::now();
    auto parser = argparse::ArgumentParser();
    configure_parser(parser);

    // Parse the command-line input against the defined parser
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &err) {
        logger.error(err.what());
        std::exit(1);
    }

    verify_arguments(parser);

    // Obtain argument values from the parsed command-line input
    std::string input_expt = parser.get<std::string>("expt");
    double param_dmin = parser.get<double>("dmin");
    bool param_force_static = parser.get<bool>("force_static");
    const int buffer_size = parser.get<int>("buffer_size");
    const std::string output_file_path = "predicted.refl";
    size_t nthreads;
    if (parser.is_used("nthreads")) {
        nthreads = parser.get<size_t>("nthreads");
    } else {
        size_t max_threads = std::thread::hardware_concurrency();
        nthreads = max_threads ? max_threads : 1;
    }

    std::ifstream f(input_expt);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error &ex) {
        logger.error("Unable to read {}; json parse error at byte {}",
                     input_expt.c_str(),
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
    if (buffer_size > 0) {
        logger.info(
          "Buffer size of {} images will be predicted either side of the scan.\n"
          "Scan static prediction is forced in this case.",
          buffer_size);
        param_force_static = true;
    }
    if (scan.get_oscillation()[1] == 0.0) {
        //For now, only implement rotation prediction.
        logger.error(
          "Data appears to be still shot, this program only implements rotation "
          "prediction");
        std::exit(1);
    }

    // Extract scan varying parameters (until we can access from the experiment object)
    scan_varying_data sv_data;
    bool scan_varying = false;
    if (!param_force_static) {
        std::tie(scan_varying, sv_data) =
          extract_scan_varying_data(elist_json_obj, scan);
    }

    if (scan_varying) {
        logger.info("Monochromatic scan-varying prediction on {}", input_expt);
    } else {
        logger.info("Monochromatic static prediction on {}", input_expt);
    }

    // Check if the minimum resolution paramenter (dmin) was passed in by the user,
    // if yes, check if it is a valid value; if not, assign a default.
    MonochromaticBeam beam = expt.beam();
    double wavelength = beam.get_wavelength();
    double dmin_min = 0.5 * wavelength;
    // FIXME: Need a better dmin_default from .expt file (like in DIALS)
    double dmin_default = dmin_min;
    if (!parser.is_used("dmin")) {
        param_dmin = dmin_default;
    } else if (param_dmin < dmin_min) {
        logger.error(
          "Prediction at a dmin of {} is not possible with wavelength {}.\n"
          "dmin must be at least 0.5 times the wavelength. Setting dmin to the\n"
          "default value of {}.\n",
          param_dmin,
          wavelength,
          dmin_default);
        param_dmin = dmin_default;
    }

    int num_reflections_initial = 0;
    int i_expt = 0;
    std::string identifier = expt.identifier();

    predicted_data_rotation output_data =
      predict_rotation(expt, sv_data, param_dmin, buffer_size, nthreads);

    // Add extra metadata to enable reflection table creation. The 'predict rotation' function can
    // be called in a loop over experiments, so here add the data which tracks which experiment
    // it came from (e.g. ids, identifiers).
    std::size_t num_new_reflections =
      output_data.panels.size() - num_reflections_initial;
    std::vector<int> new_ids(num_new_reflections, i_expt);
    // Add ids outside of per reflection loop.
    output_data.ids.insert(output_data.ids.end(), new_ids.begin(), new_ids.end());
    output_data.experiment_ids.push_back(i_expt);
    output_data.identifiers.push_back(identifier);

    // now write a reflection table.
    std::size_t sz = output_data.panels.size();
    ReflectionTable predicted(output_data.experiment_ids, output_data.identifiers);
    predicted.add_column("miller_index", sz, 3, output_data.hkl);
    predicted.add_column("panel", sz, 1, output_data.panels);
    predicted.add_column("entering", sz, 1, output_data.enter);
    predicted.add_column("s1", sz, 3, output_data.s1);
    predicted.add_column("xyzcal.px", sz, 3, output_data.xyz_px);
    predicted.add_column("xyzcal.mm", sz, 3, output_data.xyz_mm);
    predicted.add_column("flags", sz, 1, output_data.flags);
    predicted.add_column("id", sz, 1, output_data.ids);

#pragma endregion

#pragma region Write to File
    // Save reflections to file
    predicted.write(output_file_path);

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    logger.info("Saved {} reflections to {}.", sz, output_file_path);
    logger.info("Total time for prediction: {:.4f}s", elapsed_time.count());
#pragma endregion
    return 0;
}
