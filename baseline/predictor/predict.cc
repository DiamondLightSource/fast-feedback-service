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
#include <future>
#include <gemmi/symmetry.hpp>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include "index_generators.cc"
#include "ray_predictors.cc"
#include "threadpool.cc"
#include "utils.cc"

using json = nlohmann::json;

using Eigen::Matrix3d;
using Eigen::Vector3d;

constexpr size_t predicted_flag = (1 << 0);

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

struct predicted_data_rotation {
    // Shape {size, 3}
    std::vector<int> hkl;
    std::vector<double> s1;
    std::vector<double> xyz_px;
    std::vector<double> xyz_mm;
    // Shape {size, 1}
    std::vector<int> panels;
    std::vector<bool> enter;
    std::vector<size_t> flags;
    std::vector<int> ids;
    std::vector<uint64_t> experiment_ids;
    std::vector<std::string> identifiers;

    void add(const std::array<int, 3> &hkl_entry,
             const std::array<double, 3> &s1_entry,
             const std::array<double, 3> &xyz_px_entry,
             const std::array<double, 3> &xyz_mm_entry,
             int panel,
             bool enter_flag,
             size_t flag) {
        hkl.insert(hkl.end(), hkl_entry.begin(), hkl_entry.end());
        s1.insert(s1.end(), s1_entry.begin(), s1_entry.end());
        xyz_px.insert(xyz_px.end(), xyz_px_entry.begin(), xyz_px_entry.end());
        xyz_mm.insert(xyz_mm.end(), xyz_mm_entry.begin(), xyz_mm_entry.end());
        panels.push_back(panel);
        enter.push_back(enter_flag);
        flags.push_back(flag);
    }
};

struct scan_varying_data {
    std::vector<Vector3d> s0_at_scan_points;
    std::vector<Matrix3d> A_at_scan_points;
    std::vector<Matrix3d> r_setting_at_scan_points;

    auto operator<=>(const scan_varying_data &) const = default;
};

// use a global output struct for writing predictions from many threads.
predicted_data_rotation output_data{};
std::mutex write_output_data_mtx;

void predict_single_image(const int image_index,
                          const scan_varying_data &sv_data,
                          const Vector3d s0,
                          const Matrix3d A,
                          const Goniometer &goniometer,
                          const Scan &scan,
                          const Detector &detector,
                          double param_dmin,
                          gemmi::GroupOps crystal_symmetry_operations) {
    bool use_mono = true;
    const Vector3d m2 = goniometer.get_rotation_axis();
    int z0 = scan.get_image_range()[0] - 1;
    // A Rotator object that generates rotations around axis m2
    const Rotator rotator(m2);
    const Matrix3d r_fixed = goniometer.get_sample_rotation();
    const Matrix3d r_setting = goniometer.get_setting_rotation();
    const double d_osc = scan.get_oscillation()[1];
    const double osc0 = scan.get_oscillation()[0];

    predicted_data_rotation output_data_this_image;
    Vector3d s0_1 =
      sv_data.s0_at_scan_points.empty() ? s0 : sv_data.s0_at_scan_points[image_index];
    Vector3d s0_2 = sv_data.s0_at_scan_points.empty()
                      ? s0
                      : sv_data.s0_at_scan_points[image_index + 1];
    Matrix3d A1 =
      sv_data.A_at_scan_points.empty() ? A : sv_data.A_at_scan_points[image_index];
    Matrix3d A2 =
      sv_data.A_at_scan_points.empty() ? A : sv_data.A_at_scan_points[image_index + 1];
    Matrix3d r_setting_1 = sv_data.r_setting_at_scan_points.empty()
                             ? r_setting
                             : sv_data.r_setting_at_scan_points[image_index];
    Matrix3d r_setting_2 = sv_data.r_setting_at_scan_points.empty()
                             ? r_setting
                             : sv_data.r_setting_at_scan_points[image_index + 1];
    // Redefine A1 and A2 to encompass all 3 rotations
    const double phi_beg = osc0 + image_index * d_osc;
    const double phi_end = phi_beg + d_osc;
    Matrix3d r_beg = rotator.rotation_matrix(phi_beg);
    Matrix3d r_end = rotator.rotation_matrix(phi_end);
    A1 = r_setting_1 * r_beg * r_fixed * A1;
    A2 = r_setting_2 * r_end * r_fixed * A2;
    ReekeIndexGenerator index_generator(
      A1, A2, crystal_symmetry_operations, s0_1, s0_2, param_dmin, use_mono);

    std::function<std::array<std::optional<Ray>, 2>(const std::array<int, 3> &)>
      predict_ray;

    // Note that using predict_ray_monochromatic_sv seems to be 2x faster than using
    // predict_ray_monochromatic_static, and we have all the inputs required, so use that...
    // Otherwise wrap in an if/else based on if any sv_data not being empty.
    predict_ray = [=](const std::array<int, 3> &index) {
        std::array<std::optional<Ray>, 2> rays;
        rays[0] = predict_ray_monochromatic_sv(
          index, A1, A2, s0_1, s0_2, param_dmin, phi_beg, d_osc);
        return rays;
    };
    // Slower static predict function,
    /*predict_ray = [=](const std::array<int, 3>& index) {
      return predict_ray_monochromatic_static(
        index, A1, r_setting_1, r_setting_1_inv, s0, m2, rotator, param_dmin, phi_beg, d_osc
      );
  };*/

    for (;;) {
        std::optional<std::array<int, 3>> index = index_generator.next();
        if (!index) break;

        // Check if a reflection occurs at the given Miller index
        // within the required resolution.
        std::array<std::optional<Ray>, 2> rays = predict_ray(index.value());
        for (std::optional<Ray> ray : rays) {
            if (!ray) continue;
            // Append the ray
            auto impact = detector.get_ray_intersection(ray->s1);
            if (!impact.has_value()) continue;
            // Get the frame that a reflection with this angle will be observed at
            double frame = z0 + (ray->angle - osc0) / d_osc;
            intersection result = impact.value();
            auto panel = result.panel_id;
            std::array<double, 3> coords_mm = {
              result.xymm[0], result.xymm[1], ray->angle * M_PI / 180};
            std::array<double, 2> xycoords_px =
              detector.panels()[panel].mm_to_px(coords_mm[0], coords_mm[1]);
            std::array<double, 3> coords_px = {xycoords_px[0], xycoords_px[1], frame};
            std::array<double, 3> s1 = {ray->s1[0], ray->s1[1], ray->s1[2]};
            output_data_this_image.add(index.value(),
                                       s1,
                                       coords_px,
                                       coords_mm,
                                       panel,
                                       ray->entering,
                                       predicted_flag);
        }
    }
    // now write to shared output data.
    std::lock_guard<std::mutex> lock(write_output_data_mtx);
    output_data.hkl.insert(output_data.hkl.end(),
                           output_data_this_image.hkl.begin(),
                           output_data_this_image.hkl.end());
    output_data.s1.insert(output_data.s1.end(),
                          output_data_this_image.s1.begin(),
                          output_data_this_image.s1.end());
    output_data.xyz_px.insert(output_data.xyz_px.end(),
                              output_data_this_image.xyz_px.begin(),
                              output_data_this_image.xyz_px.end());
    output_data.xyz_mm.insert(output_data.xyz_mm.end(),
                              output_data_this_image.xyz_mm.begin(),
                              output_data_this_image.xyz_mm.end());
    output_data.panels.insert(output_data.panels.end(),
                              output_data_this_image.panels.begin(),
                              output_data_this_image.panels.end());
    output_data.enter.insert(output_data.enter.end(),
                             output_data_this_image.enter.begin(),
                             output_data_this_image.enter.end());
    output_data.flags.insert(output_data.flags.end(),
                             output_data_this_image.flags.begin(),
                             output_data_this_image.flags.end());
}

void predict_rotation(Experiment<MonochromaticBeam> &experiment,
                      const scan_varying_data &sv_data,
                      double param_dmin,
                      int buffer_size = 0,
                      int nthreads = 1) {
    Scan &scan = experiment.scan();
    if (buffer_size > 0) {
        // Can't have both scan varying data and a buffer
        // (as the models will not be defined outside the scan range)
        if (sv_data != scan_varying_data{}) {
            throw std::runtime_error(
              "Can't call predict function with scan varying data and an image "
              "buffer.");
        }
        std::array<int, 2> image_range = {scan.get_image_range()[0] - buffer_size,
                                          scan.get_image_range()[1] + buffer_size};
        std::array<double, 2> oscillation = {
          scan.get_oscillation()[0] - buffer_size * scan.get_oscillation()[1],
          scan.get_oscillation()[1]};
        scan = Scan(image_range, oscillation);
    }
    MonochromaticBeam &beam = experiment.beam();
    const Goniometer &goniometer = experiment.goniometer();
    Detector &detector = experiment.detector();
    const Crystal &crystal = experiment.crystal();

    gemmi::GroupOps crystal_symmetry_operations =
      crystal.get_space_group().operations();
    const Matrix3d A = crystal.get_A_matrix();
    const Vector3d m2 = goniometer.get_rotation_axis();
    // A Rotator object that generates rotations around axis m2
    const Rotator rotator(m2);
    const Matrix3d r_fixed = goniometer.get_sample_rotation();
    const Matrix3d r_setting = goniometer.get_setting_rotation();
    const double d_osc = scan.get_oscillation()[1];
    const double osc0 = scan.get_oscillation()[0];

    int z0 = scan.get_image_range()[0] - 1;
    int z1 = scan.get_image_range()[1];

    Vector3d s0 = beam.get_s0();
    int n_images = z1 - z0;
    nthreads = std::min(n_images, nthreads);
    logger.info("Predicting spots on {} images over {} threads", n_images, nthreads);

    ThreadPool pool(nthreads);

    for (int image_index = 0; image_index < n_images; ++image_index) {
        pool.enqueue([=] {
            predict_single_image(image_index,
                                 sv_data,
                                 s0,
                                 A,
                                 goniometer,
                                 scan,
                                 detector,
                                 param_dmin,
                                 crystal_symmetry_operations);
        });
    }
}

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
        const int num_images =
          scan.get_image_range()[1] - scan.get_image_range()[0] + 1;
        json beam_data = elist_json_obj.at("beam")[0];
        json goniometer_data = elist_json_obj.at("goniometer")[0];
        json crystal_data = elist_json_obj.at("crystal")[0];
        json s0_at_scan_points;
        json A_at_scan_points;
        json r_setting_at_scan_points;
        if (beam_data.contains("s0_at_scan_points")) {
            scan_varying = true;
            s0_at_scan_points = beam_data.at("s0_at_scan_points");
            if (s0_at_scan_points.size()
                == num_images + 1) {  // i.e. is expected length.
                std::vector<Vector3d> scan_varying_s0;
                for (const auto &entry : s0_at_scan_points) {
                    Vector3d vec(entry[0].get<double>(),
                                 entry[1].get<double>(),
                                 entry[2].get<double>());
                    scan_varying_s0.push_back(vec);
                }
                sv_data.s0_at_scan_points = scan_varying_s0;
            }
        }
        if (crystal_data.contains("A_at_scan_points")) {
            scan_varying = true;
            A_at_scan_points = crystal_data.at("A_at_scan_points");
            if (A_at_scan_points.size() == num_images + 1) {
                std::vector<Matrix3d> scan_varying_A;
                for (const auto &entry : A_at_scan_points) {
                    Matrix3d A_mat;
                    A_mat << entry[0].get<double>(), entry[1].get<double>(),
                      entry[2].get<double>(), entry[3].get<double>(),
                      entry[4].get<double>(), entry[5].get<double>(),
                      entry[6].get<double>(), entry[7].get<double>(),
                      entry[8].get<double>();
                    scan_varying_A.push_back(A_mat);
                }
                sv_data.A_at_scan_points = scan_varying_A;
            }
        }
        if (goniometer_data.contains("setting_rotation_at_scan_points")) {
            scan_varying = true;
            r_setting_at_scan_points =
              goniometer_data.at("setting_rotation_at_scan_points");
            if (r_setting_at_scan_points.size() == num_images + 1) {
                std::vector<Matrix3d> scan_varying_r;
                for (const auto &entry : r_setting_at_scan_points) {
                    Matrix3d r_mat;
                    r_mat << entry[0].get<double>(), entry[1].get<double>(),
                      entry[2].get<double>(), entry[3].get<double>(),
                      entry[4].get<double>(), entry[5].get<double>(),
                      entry[6].get<double>(), entry[7].get<double>(),
                      entry[8].get<double>();
                    scan_varying_r.push_back(r_mat);
                }
                sv_data.r_setting_at_scan_points = scan_varying_r;
            }
        }
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
