/**
 * @file baseline_integrator.cc
 * @brief Main application for baseline CPU-only integration
 */

#include <Eigen/Dense>
#include <dx2/beam.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>
#include <filesystem>
#include <fstream>
#include <gemmi/unitcell.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "arg_parser.hpp"
#include "background.cc"
#include "common.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"
#include "integration_calculations.cc"
#include "integrator/extent.hpp"
#include "kabsch.cc"
#include "math/math_utils.cuh"
#include "predict.cc"
#include "sigma_estimation.cc"
#include "version.hpp"

using json = nlohmann::json;
using Eigen::Vector3i;

constexpr size_t IntegratedSum = (1 << 8);
constexpr double DEG2RAD = M_PI / 180.0;
constexpr double RAD2DEG = 180.0 / M_PI;


enum class FGAlgorithm : uint8_t {
    Ellipsoid,
    Dials
};


#pragma region Argument Parsing
class BaselineIntegratorArgumentParser : public FFSArgumentParser {
  public:
    BaselineIntegratorArgumentParser(std::string version) : FFSArgumentParser(version) {
        add_h5read_arguments();
        add_integrator_arguments();
    }

    void add_h5read_arguments() override {
        add_argument("reflection")
          .metavar("strong.refl")
          .help("Input reflection table")
          .action([&](const std::string &value) { _reflection_filepath = value; });

        add_argument("experiment")
          .metavar("experiments.expt")
          .help("Input experiment list")
          .action([&](const std::string &value) { _experiment_filepath = value; });
    }

    auto const reflections() const -> std::string {
        return _reflection_filepath;
    }
    auto const experiment() const -> std::string {
        return _experiment_filepath;
    }

  private:
    std::string _reflection_filepath;
    std::string _experiment_filepath;

    void add_integrator_arguments() {
        add_argument("--timeout")
          .help("Amount of time (in seconds) to wait for new images before failing.")
          .metavar("S")
          .default_value<float>(30.0f)
          .scan<'f', float>();

        // Taken in as degrees, converted to radians later
        add_argument("--sigma_m")
          .help(
            "Sigma_m (deg): Standard deviation of the rotation axis in reciprocal "
            "space.")
          .metavar("σm")
          .scan<'f', float>();

        // Taken in as degrees, converted to radians later
        add_argument("--sigma_b")
          .help(
            "Sigma_b (deg): Standard deviation of the beam direction in reciprocal "
            "space.")
          .metavar("σb")
          .scan<'f', float>();

        add_argument("--sigma_estimation.min_bbox_depth", "--min_bbox_depth")
          .help(
            "When calculating sigma_m, only use reflections that span at least this "
            "number of images.")
          .default_value<int>(6)
          .scan<'i', int>();

        add_argument("-a", "--algorithm")
          .help("Foreground algorithm choice - dials or ellipsoid.")
          .default_value<std::string>("ellipsoid");

        add_argument("--min_zeta")
          .help(
            "Reflections close to the rotation axis, with a zeta below this threshold "
            "will not be integrated.")
          .default_value<float>(0.05)
          .scan<'f', float>();

        add_argument("output")
          .help("Output file path")
          .metavar("integrated.refl")
          .default_value<std::string>("integrated.refl");

        add_argument("--nthreads")  // mainly for testing.
          .help(
            "The number of threads to use for the image integration loop."
            "Defaults to the value of std::thread::hardware_concurrency.")
          .scan<'u', size_t>();
    }
};
#pragma endregion Argument Parsing

struct Accumulator {
    std::vector<double> intensity;
    std::vector<size_t> nfg;
    std::vector<size_t> nbg;
    std::vector<Vector3d> intensity_times_coord;
    std::vector<BackgroundAggregator> bg;
    std::vector<bool> success;

    Accumulator(size_t n_reflections)
      : intensity(n_reflections, 0.0),
        nfg(n_reflections, 0),
        nbg(n_reflections, 0),
        intensity_times_coord(n_reflections, Vector3d::Zero()),
        bg(n_reflections),
        success(n_reflections, true) {}

};


struct ImageRange {
    size_t begin;
    size_t end;   // half-open [begin, end)
};

std::vector<ImageRange> split_ranges(size_t n_images, size_t n_threads) {
    std::vector<ImageRange> ranges(n_threads);

    size_t base = n_images / n_threads;
    size_t rem  = n_images % n_threads;

    size_t current = 0;
    for (size_t t = 0; t < n_threads; t++) {
        size_t count = base + (t < rem ? 1 : 0);
        ranges[t] = {current, current + count};
        current += count;
    }

    return ranges;
}

struct SharedConstants {
    // --- Sizes ---
    size_t n_reflections;

    // --- Reflection selection ---
    const std::vector<bool>& dont_integrate;
    const std::vector<BoundingBoxExtents>& computed_bounding_boxes;
    const std::vector<CoordinateSystem>& coord_system_vector;

    // --- Reflection geometry ---
    const mdspan_type<double> phi_column;     // shape: [n_reflections, 3]
    const mdspan_type<double> s1_vectors;     // shape: [n_reflections, 3]

    // --- Panel / detector ---
    const Panel& panel;

    // --- Scan / rotation ---
    double phi0;
    double dphi;
    double DEG2RAD;

    // --- Beam ---
    double s0_length;

    // --- Foreground/background model ---
    FGAlgorithm fg_algorithm;   // "ellipsoid" or "dials"

    double delta_b_r2;
    double delta_m_r2;

    SharedConstants(
        size_t n_reflections_,
        const std::vector<bool>& dont_integrate_,
        const std::vector<BoundingBoxExtents>& computed_bounding_boxes_,
        const std::vector<CoordinateSystem>& coord_system_vector_,
        const mdspan_type<double> phi_column_,
        const mdspan_type<double> s1_vectors_,
        const Panel& panel_,
        double phi0_,
        double dphi_,
        double s0_length_,
        FGAlgorithm fg_algorithm_,
        double delta_b_r2_,
        double delta_m_r2_
    )
      : n_reflections(n_reflections_),
        dont_integrate(dont_integrate_),
        computed_bounding_boxes(computed_bounding_boxes_),
        coord_system_vector(coord_system_vector_),
        phi_column(phi_column_),
        s1_vectors(s1_vectors_),
        panel(panel_),
        phi0(phi0_),
        dphi(dphi_),
        DEG2RAD(M_PI / 180.0),
        s0_length(s0_length_),
        fg_algorithm(std::move(fg_algorithm_)),
        delta_b_r2(delta_b_r2_),
        delta_m_r2(delta_m_r2_) {}
};


Accumulator process_image_range(
    ImageRange range,
    Experiment<MonochromaticBeam>& expt,
    std::unordered_map<int, std::vector<size_t>>& reflections_by_image,
    const SharedConstants& constants
) {
    Accumulator accumulator(constants.n_reflections);

    std::string filename = expt.imagesequence().filename();
    H5Read reader(filename);

    auto [image_slow, image_fast] = reader.image_shape();
    auto mask = reader.get_mask().value();

    for (size_t j = range.begin; j < range.end; j++) {
        if (!reader.is_image_available(j))
            continue;

        auto image = reader.get_image(j);
        logger.info("Processing image {}", j + 1);

        std::vector<double> dxy_start;// start of image (in rotation)
        std::vector<double> dxy_end;// end of image (in rotation)
        std::vector<double> dxy;
        double phidash_start = (constants.phi0 + (j * constants.dphi)) * constants.DEG2RAD;
        double phidash_end = (constants.phi0 + ((j + 1) * constants.dphi)) * constants.DEG2RAD;

        switch (constants.fg_algorithm){
        case FGAlgorithm::Ellipsoid:

            for (size_t k = 0; k < reflections_by_image[j].size(); k++) {
                size_t refl_id = reflections_by_image[j][k];
                if (constants.dont_integrate[refl_id]) {
                    continue;
                }
                const auto &bbox = constants.computed_bounding_boxes[refl_id];
                
                // calculate dxy array (arrays of distances in kabsch space from pixel corners to xyzcal)
                int n_x = bbox.x_max - bbox.x_min + 1;
                int n_y = bbox.y_max - bbox.y_min + 1;
                //std::vector<double> dxy_start(n_x
                //                              * n_y);    // start of image (in rotation)
                //std::vector<double> dxy_end(n_x * n_y);  // end of image (in rotation)

                size_t required = static_cast<size_t>(n_x) * n_y;
                if (dxy_start.size() < required) {
                    dxy_start.resize(required);
                    dxy_end.resize(required);
                }

                
                double phi_c = constants.phi_column(refl_id, 2);
                CoordinateSystem cs = constants.coord_system_vector[refl_id];
                //Vector3d s1_this = Eigen::Map<Vector3d>(&constants.s1_vectors(refl_id, 0));
                
                for (int y = 0; y < n_y; ++y) {
                    int row = y * n_x;
                    for (int x = 0; x < n_x; ++x) {
                        int index = row + x;


                        //for (int x = 0; x < n_x; x++) {
                        //for (int y = 0; y < n_y; y++) {
                        //int index = x + (y * (n_x));
                        std::array<double, 2> px_mm =
                          constants.panel.px_to_mm(x + bbox.x_min, y + bbox.y_min);
                        Vector3d s1_ = constants.panel.get_lab_coord(px_mm[0], px_mm[1]);
                        s1_.normalize();
                        Vector3d s1dash = s1_ * constants.s0_length;
                        Vector3d epsilon_coords_start =
                          cs.coords_from_s1vector(s1dash, phidash_start);
                        Vector3d epsilon_coords_end =
                          cs.coords_from_s1vector(s1dash, phidash_end);
                        if ((phidash_start > phi_c) && (phi_c < phidash_end)) {
                            epsilon_coords_start[2] = 0.0;
                            epsilon_coords_end[2] = 0.0;
                        }

                        dxy_start[index] =
                          ((epsilon_coords_start[0] * epsilon_coords_start[0]
                            + epsilon_coords_start[1] * epsilon_coords_start[1])
                           * constants.delta_b_r2)
                          + ((epsilon_coords_start[2] * epsilon_coords_start[2])
                             * constants.delta_m_r2);
                        dxy_end[index] =
                          ((epsilon_coords_end[0] * epsilon_coords_end[0]
                            + epsilon_coords_end[1] * epsilon_coords_end[1])
                           * constants.delta_b_r2)
                          + ((epsilon_coords_end[2] * epsilon_coords_end[2])
                             * constants.delta_m_r2);
                    }
                }
                // Now loop through pixels in bbox, adding to foreground or background if unmasked and in image.
                for (int x = bbox.x_min; x < bbox.x_max; x++) {
                    for (int y = bbox.y_min; y < bbox.y_max; y++) {
                        // The bounding box may extend beyond the image - if so, this is ok if it is only background
                        // but not if it is foreground.
                        int x_index = x - bbox.x_min;
                        int y_index = y - bbox.y_min;
                        int dxyindex = x_index + (y_index * n_x);
                        double d1 = dxy_start[dxyindex];
                        double d2 = dxy_start[dxyindex + 1];
                        double d3 = dxy_start[dxyindex + n_x];
                        double d4 = dxy_start[dxyindex + n_x + 1];
                        double d5 = dxy_end[dxyindex];
                        double d6 = dxy_end[dxyindex + 1];
                        double d7 = dxy_end[dxyindex + n_x];
                        double d8 = dxy_end[dxyindex + n_x + 1];
                        double d = d1;
                        d = std::min(d, d2);
                        d = std::min(d, d3);
                        d = std::min(d, d4);
                        d = std::min(d, d5);
                        d = std::min(d, d6);
                        d = std::min(d, d7);
                        d = std::min(d, d8);
                        /*double d =
                          std::min(std::min(std::min(d1, d2), std::min(d3, d4)),
                                   std::min(std::min(d5, d6), std::min(d7, d8)));*/
                        int index = x + (y * image_fast);
                        bool in_image =
                          x >= 0 && x < image_fast && y >= 0 && y < image_slow;
                        if (d <= 1.0) {
                            // Foreground
                            if (in_image) {
                                if (mask[index] == 0) {  // masked
                                    accumulator.success[refl_id] = false;
                                } else {
                                    auto data = image.data.data()[index];
                                    accumulator.intensity[refl_id] += data;
                                    accumulator.nfg[refl_id] += 1;
                                    Vector3d I_times_c = {data * (x + 0.5),
                                                          data * (y + 0.5),
                                                          data * (j + 0.5)};
                                    accumulator.intensity_times_coord[refl_id] +=
                                      I_times_c;
                                }
                            } else {
                                accumulator.success[refl_id] = false;
                            }
                        } else {  // d>1.0
                            // Background
                            if (in_image) {
                                if (mask[index] != 0) {
                                    // In image, not masked
                                    auto data = image.data.data()[index];
                                    accumulator.bg[refl_id].add(data);
                                    accumulator.nbg[refl_id] += 1;
                                }
                            }
                        }
                    }
                }
            }
            break;

        case FGAlgorithm::Dials:

            for (size_t k = 0; k < reflections_by_image[j].size(); k++) {
                size_t refl_id = reflections_by_image[j][k];
                if (constants.dont_integrate[refl_id]) {
                    continue;
                }
                const auto &bbox = constants.computed_bounding_boxes[refl_id];
                // dials algorithm - a fixed ellipse (2D) across all images in the bbox
                int n_x = bbox.x_max - bbox.x_min + 1;
                int n_y = bbox.y_max - bbox.y_min + 1;
                //std::vector<double> dxy(n_x * n_y);
                size_t required = static_cast<size_t>(n_x) * n_y;
                if (dxy.size() < required) {
                    dxy.resize(required);
                }
                double phi =
                  constants.phi0
                  + (j * constants.dphi)
                      * constants.DEG2RAD;  // Required for calls but not actually used as don't use e3.
                CoordinateSystem cs = constants.coord_system_vector[refl_id];
                //Vector3d s1_this = Eigen::Map<Vector3d>(&constants.s1_vectors(refl_id, 0));
                for (int x = 0; x < n_x; x++) {
                    for (int y = 0; y < n_y; y++) {
                        int index = x + (y * (n_x));
                        std::array<double, 2> px_mm =
                          constants.panel.px_to_mm(x + bbox.x_min, y + bbox.y_min);
                        Vector3d s1_ = constants.panel.get_lab_coord(px_mm[0], px_mm[1]);
                        s1_.normalize();
                        Vector3d s1dash = s1_ * constants.s0_length;
                        Vector3d epsilon_coords = cs.coords_from_s1vector(s1dash, phi);

                        dxy[index] = ((epsilon_coords[0] * epsilon_coords[0]
                                       + epsilon_coords[1] * epsilon_coords[1])
                                      * constants.delta_b_r2);
                    }
                }
                // Now loop through pixels in bbox, adding to foreground or background if unmasked and in image.
                for (int x = bbox.x_min; x < bbox.x_max; x++) {
                    for (int y = bbox.y_min; y < bbox.y_max; y++) {
                        // The bounding box may extend beyond the image - if so, this is ok if it iis only background
                        // but not if it is foreground.
                        int x_index = x - bbox.x_min;
                        int y_index = y - bbox.y_min;
                        int dxyindex = x_index + (y_index * n_x);
                        double d1 = dxy[dxyindex];
                        double d2 = dxy[dxyindex + 1];
                        double d3 = dxy[dxyindex + n_x];
                        double d4 = dxy[dxyindex + n_x + 1];
                        double d = d1;
                        d = std::min(d, d2);
                        d = std::min(d, d3);
                        d = std::min(d, d4);
                        //double d = std::min(std::min(d1, d2), std::min(d3, d4));
                        int index = x + (y * image_fast);
                        bool in_image =
                          x >= 0 && x < image_fast && y >= 0 && y < image_slow;

                        if (d <= 1.0) {
                            // Foreground
                            if (in_image) {
                                if (mask[index] == 0) {  // masked)
                                    accumulator.success[refl_id] = false;
                                } else {
                                    auto data = image.data.data()[index];
                                    accumulator.intensity[refl_id] += data;
                                    accumulator.nfg[refl_id] += 1;
                                    Vector3d I_times_c = {data * (x + 0.5),
                                                          data * (y + 0.5),
                                                          data * (j + 0.5)};
                                    accumulator.intensity_times_coord[refl_id] +=
                                      I_times_c;
                                }
                            } else {
                                accumulator.success[refl_id] = false;
                            }
                        } else {  // d>1.0
                            // Background
                            if (in_image) {
                                if (mask[index] != 0) {
                                    // In image, not masked
                                    auto data = image.data.data()[index];
                                    accumulator.bg[refl_id].add(data);
                                    accumulator.nbg[refl_id] += 1;
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
    }
    return accumulator;
}

FGAlgorithm parse_fg_algorithm(const std::string& name) {
    if (name == "ellipsoid") return FGAlgorithm::Ellipsoid;
    if (name == "dials")     return FGAlgorithm::Dials;
    throw std::runtime_error("Unknown foreground algorithm: " + name);
}


#pragma region Application Entry
int main(int argc, char **argv) {
    logger.info("Baseline Integrator Version: {}", FFS_VERSION);

    // Parse arguments
    auto parser = BaselineIntegratorArgumentParser(FFS_VERSION);
    auto args = parser.parse_args(argc, argv);
    const auto reflection_file = parser.reflections();
    const auto experiment_file = parser.experiment();

    float timeout = parser.get<float>("timeout");
    std::string output_file = parser.get<std::string>("output");
    
    FGAlgorithm fg_algorithm = parse_fg_algorithm(parser.get<std::string>("algorithm"));

    // Guard against missing files
    if (!std::filesystem::exists(reflection_file)) {
        logger.error("Reflection file not found: {}", reflection_file);
        return 1;
    }
    if (!std::filesystem::exists(experiment_file)) {
        logger.error("Experiment file not found: {}", experiment_file);
        return 1;
    }

    // Load reflection data.
    // Expectation is that it will be indexed/refined reflection data, with the sigma_b_variance,
    // sigma_m_variance and spot_extent_z parameters (if processed with the ffs spotfinder), which will
    // be used to calculate sigma_b and sigma_m. Then prediction code will be run as part of integration.
    // The input can also be predicted reflection data - in that case we will have to supply
    // values for sigma_b, sigma_m but won't need to call predict code again.
    logger.info("Loading reflection data from: {}", reflection_file);
    ReflectionTable reflections(reflection_file);

    // Display column names for debugging
    std::string column_names_str;
    for (const auto &name : reflections.get_column_names()) {
        column_names_str += "\n\t- " + name;
    }
    logger.info("Available columns: {}", column_names_str);

    // Parse experiment list from JSON
    logger.info("Loading experiment data from: {}", experiment_file);
    std::ifstream f(experiment_file);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error &ex) {
        logger.error("Failed to parse experiment file '{}': byte {}, {}",
                     experiment_file,
                     ex.byte,
                     ex.what());
        return 1;
    }

    // Construct Experiment object and extract components
    Experiment<MonochromaticBeam> expt;
    try {
        expt = Experiment<MonochromaticBeam>(elist_json_obj);
    } catch (const std::invalid_argument &ex) {
        logger.error(
          "Failed to construct Experiment from '{}': {}", experiment_file, ex.what());
        return 1;
    }

    // Extract experimental components
    MonochromaticBeam beam = expt.beam();
    Goniometer gonio = expt.goniometer();
    const Panel &panel = expt.detector().panels()[0];  // Assuming single panel detector
    const Scan &scan = expt.scan();
    const Crystal &crystal = expt.crystal();

    // Extract vectors and parameters
    Eigen::Vector3d s0 = beam.get_s0();
    Eigen::Vector3d rotation_axis = gonio.get_rotation_axis();
    const auto [osc_start, osc_width] = scan.get_oscillation();
    int image_range_start = scan.get_image_range()[0];
    int image_range_end = scan.get_image_range()[1];
    double wl = beam.get_wavelength();
    gemmi::UnitCell cell = crystal.get_unit_cell();

    logger.info("Experimental parameters:");
    logger.info("  Beam s0: ({:.6f}, {:.6f}, {:.6f})", s0.x(), s0.y(), s0.z());
    logger.info("  Wavelength: {:.6f} Å", wl);
    logger.info("  Rotation axis: ({:.6f}, {:.6f}, {:.6f})",
                rotation_axis.x(),
                rotation_axis.y(),
                rotation_axis.z());
    logger.info("  Oscillation: start={:.3f}°, width={:.3f}°", osc_start, osc_width);
    logger.info("  Image range: {} to {}", image_range_start, image_range_end);

#pragma region Sigma estimation
    // If input is a predicted refl, then we require sigma_b, sigma_m as we will not be
    // able to estimate it from the data
    // Else the input as an indexed.refl/refined.refl with the sigma variance columns,
    // then we can calculate sigma_b, sigma_m but will have to also run the predict
    // code in this program.
    float sigma_b = 0.0;
    float sigma_m = 0.0;
    float min_zeta = parser.get<float>("min_zeta");
    if (parser.is_used("sigma_m")) {
        sigma_m = degrees_to_radians(
          parser.get<float>("sigma_m"));  // Use radians for calculations
    }
    if (parser.is_used("sigma_b")) {
        sigma_b = degrees_to_radians(
          parser.get<float>("sigma_b"));  // Use radians for calculations
    }

    // Estimate sigmas
    auto sigma_b_data = reflections.column<double>("sigma_b_variance");
    auto sigma_m_data = reflections.column<double>("sigma_m_variance");
    auto extent_z_data = reflections.column<int>("spot_extent_z");
    if (sigma_b_data && sigma_m_data && extent_z_data) {
        int min_bbox_depth = parser.get<int>("sigma_estimation.min_bbox_depth");
        // Estimate the values from the data, and use if user hasn't specified values.
        auto [sigma_b_calc, sigma_m_calc] =
          estimate_sigmas(reflections, expt, min_bbox_depth);
        if (sigma_m == 0.0) {
            sigma_m = sigma_m_calc;
        }
        if (sigma_b == 0.0) {
            sigma_b = sigma_b_calc;
        }
    }
    if (sigma_b == 0.0) {
        logger.error(
          "No value for sigma_b. This must either be provided as input, or an input "
          "reflection "
          "table containing sigma_b_variance must be used.");
        return 1;
    }
    if (sigma_m == 0.0) {
        logger.error(
          "No value for sigma_m. This must either be provided as input, or an input "
          "reflection "
          "table containing sigma_m_variance and spot_extent_z must be used.");
        return 1;
    }
#pragma endregion Sigma estimation

#pragma region Predict or extract predictions
    // Determine if the data are predicted based on the reflection flags
    // If data are not predicted, run predict code.
    auto flags_column_opt = reflections.column<size_t>("flags");
    if (!flags_column_opt) {
        logger.error("Column 'flags' not found.");
        return 1;
    }
    auto flags = *flags_column_opt;
    bool all_predicted = true;
    for (int i = 0; i < flags.extent(0); ++i) {
        auto f = flags(i, 0);
        if (!(f & predicted_flag)) {
            all_predicted = false;
            break;
        }
    }
    if (all_predicted) {
        logger.info(
          "Input data have the predict flag set, treating as predicted data.");
    }

    size_t nthreads;
    if (parser.is_used("nthreads")) {
        nthreads = parser.get<size_t>("nthreads");
    } else {
        size_t max_threads = std::thread::hardware_concurrency();
        nthreads = max_threads ? max_threads : 1;
    };

    mdspan_type<double> phi_column;
    mdspan_type<double> s1_vectors;
    std::vector<int> hkl_vectors;
    size_t num_reflections;
    predicted_data_rotation output_data;  // Define here so that members stay in scope
    if (!all_predicted) {
        scan_varying_data sv_data;
        bool scan_varying = false;
        std::tie(scan_varying, sv_data) =
          extract_scan_varying_data(elist_json_obj, scan);
        if (scan_varying) {
            logger.info("Monochromatic scan-varying prediction");
        } else {
            logger.info("Monochromatic static prediction");
        }

        double wavelength = beam.get_wavelength();
        double dmin_min = 0.5 * wavelength;
        // FIXME: Need a better dmin_default from .expt file (like in DIALS)
        double dmin_default = dmin_min;
        double param_dmin = dmin_default;
        int buffer_size = 0;
        output_data =
          predict_rotation(expt, sv_data, param_dmin, buffer_size, nthreads);
        std::size_t num_new_reflections = output_data.panels.size();
        logger.info("Predicted {} reflections", num_new_reflections);

        s1_vectors =
          mdspan_type<double>(output_data.s1.data(), output_data.s1.size() / 3, 3);
        phi_column = mdspan_type<double>(
          output_data.xyz_mm.data(), output_data.xyz_mm.size() / 3, 3);
        num_reflections = output_data.enter.size();
        hkl_vectors = output_data.hkl;
    } else {
        // is already predicted, so just extract required data.
        auto s1_vectors_opt = reflections.column<double>("s1");
        if (!s1_vectors_opt) {
            logger.error("Column 's1' not found in reflection data.");
            return 1;
        }
        s1_vectors = *s1_vectors_opt;
        auto phi_column_opt = reflections.column<double>("xyzcal.mm");
        if (!phi_column_opt) {
            logger.error("Column 'xyzcal.mm' not found for phi positions.");
            return 1;
        }
        auto hkl_vectors_opt = reflections.column<int>("miller_index");
        if (!hkl_vectors_opt) {
            logger.error("Column 'miller_index' not found in reflection data.");
            return 1;
        }
        phi_column = *phi_column_opt;
        num_reflections = s1_vectors.extent(0);
        hkl_vectors = std::vector<int>(
          hkl_vectors_opt.value().data_handle(),
          hkl_vectors_opt.value().data_handle() + hkl_vectors_opt.value().size());
    }

#pragma endregion Predict or extract predictions

#pragma region Calculate or extract bounding boxes
    // Create a new reflection table for the output.
    std::vector<std::string> identifiers = reflections.get_identifiers();
    std::vector<uint64_t> ids = reflections.get_experiment_ids();
    ReflectionTable integrated_data(ids, identifiers);

    logger.info("Processing {} reflections", num_reflections);

    // Compute bounding boxes using baseline CPU algorithms
    auto computed_bounding_boxes_opt = reflections.column<int>("bbox");
    std::vector<BoundingBoxExtents> computed_bounding_boxes;
    // For testing, we want to be able to load data with calculated bboxes
    // But indexed.refl data can also have bbox from spotfinding. So load
    // and check the length compared to size of predictions
    bool need_to_calc_bbox = true;
    if (computed_bounding_boxes_opt.has_value()) {
        int n_refls = computed_bounding_boxes_opt.value().size() / 6;
        if (n_refls == num_reflections) {
            need_to_calc_bbox = false;
        }
    }
    if (need_to_calc_bbox) {
        logger.info("Computing Kabsch bounding boxes using baseline CPU algorithms...");
        computed_bounding_boxes = compute_kabsch_bounding_boxes(s0,
                                                                rotation_axis,
                                                                s1_vectors,
                                                                phi_column,
                                                                num_reflections,
                                                                sigma_b,
                                                                sigma_m,
                                                                panel,
                                                                scan,
                                                                beam);

        logger.info("Successfully computed {} bounding boxes",
                    computed_bounding_boxes.size());
    } else {
        std::vector<int> bbox_data =
          std::vector<int>(computed_bounding_boxes_opt.value().data_handle(),
                           computed_bounding_boxes_opt.value().data_handle()
                             + computed_bounding_boxes_opt.value().size());
        for (int i = 0; i < bbox_data.size(); i += 6) {
            BoundingBoxExtents bbox;
            bbox.x_min = bbox_data[i];
            bbox.x_max = bbox_data[i + 1];
            bbox.y_min = bbox_data[i + 2];
            bbox.y_max = bbox_data[i + 3];
            bbox.z_min = bbox_data[i + 4];
            bbox.z_max = bbox_data[i + 5];
            computed_bounding_boxes.push_back(bbox);
        }
        logger.info("Successfully loaded {} bounding boxes from input reflections",
                    computed_bounding_boxes.size());
    }

#pragma endregion Calculate or extract bounding boxes

#pragma region Map reflections to images
    // Map reflections by z layer (image number)
    logger.info("Mapping reflections by image number (z layer)");
    std::unordered_map<int, std::vector<size_t>> reflections_by_image;
    for (size_t refl_id = 0; refl_id < num_reflections; ++refl_id) {
        const auto &bbox = computed_bounding_boxes[refl_id];
        // Add this reflection to all images it spans
        for (int z = bbox.z_min; z < bbox.z_max; ++z) {
            reflections_by_image[z].push_back(refl_id);
        }
    }
    logger.info("Reflections mapped across {} unique images",
                reflections_by_image.size());

    std::vector<int> keys;
    keys.reserve(reflections_by_image.size());
    for (const auto &kv : reflections_by_image) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());

    for (int image : keys) {
        const auto &refl_list = reflections_by_image.at(image);
        logger.info("Image {} has {} reflections", image + 1, refl_list.size());
    }
#pragma endregion Map reflections to images

#pragma region Loop over images and perform summation
    // Set up data arrays to fill during accumulation
    std::vector<int> intensity_accumulators(num_reflections);
    std::vector<Vector3d> intensity_times_coord_accumulators(num_reflections);
    std::vector<BackgroundAggregator> bg_accumulators(num_reflections);
    std::vector<int> nbg_accumulators(num_reflections);
    std::vector<int> nfg_accumulators(num_reflections);
    std::vector<bool> success(num_reflections, true);
    std::vector<CoordinateSystem> coord_system_vector =
      {};  // a vector of coordinate systems for each reflection.
    coord_system_vector.reserve(num_reflections);
    // Precalculate a few things
    double s0_length = beam.get_s0().norm();
    double phi0 = scan.get_oscillation()[0];
    double dphi = scan.get_oscillation()[1];
    const Vector3d m2 = gonio.get_rotation_axis();
    std::vector<bool> dont_integrate(num_reflections, false);
    for (int i = 0; i < num_reflections; i++) {
        Vector3d s1_this = Eigen::Map<Vector3d>(&s1_vectors(i, 0));
        CoordinateSystem cs(m2, s0, s1_this, phi_column(i, 2));
        coord_system_vector.push_back(cs);
        if (std::abs(cs.zeta()) < min_zeta) {
            dont_integrate[i] = true;
            success[i] = false;
        }
    }

    double delta_b = sigma_b * 3.0;
    double delta_b_r2 = 1.0 / (delta_b * delta_b);
    double delta_m = sigma_m * 3.0;
    double delta_m_r2 = 1.0 / (delta_m * delta_m);

    logger.info("Calculated coordinate systems");

    // Now load some image data and loop through:
    std::string filename = expt.imagesequence().filename();
    auto reader = H5Read(filename);
    size_t n_images = reader.get_number_of_images();
    auto [image_slow, image_fast] = reader.image_shape();
    auto mask = reader.get_mask().value();

    SharedConstants constants = SharedConstants(
        num_reflections, dont_integrate,
        computed_bounding_boxes,
        coord_system_vector,
        phi_column, s1_vectors, panel,
        phi0, dphi, s0_length, fg_algorithm, delta_b_r2, delta_m_r2);
    logger.info("Made shared constants");

    std::vector<ImageRange> ranges = split_ranges(n_images, nthreads);
    std::vector<std::thread> workers;
    //std::vector<Accumulator> thread_accs(nthreads);

    std::vector<Accumulator> thread_accs;
    thread_accs.reserve(nthreads);

    for (size_t t = 0; t < nthreads; t++) {
        thread_accs.emplace_back(num_reflections);
    }

    logger.info("About to start parallelisation");
    for (size_t t = 0; t < nthreads; t++) {
        workers.emplace_back([&, t] {
            thread_accs[t] =
                std::move(process_image_range(ranges[t], expt, reflections_by_image, constants));
        });
    }

    for (auto& th : workers)
        th.join();

    for (const auto& acc : thread_accs) {
        for (size_t r = 0; r < num_reflections; r++) {
            intensity_accumulators[r] += acc.intensity[r];
            nfg_accumulators[r] += acc.nfg[r];
            nbg_accumulators[r] += acc.nbg[r];
            bg_accumulators[r].add(acc.bg[r]);
            intensity_times_coord_accumulators[r] += acc.intensity_times_coord[r];
            success[r] = static_cast<bool>(success[r]) && acc.success[r];
        }
    }


    

    /*Accumulator accumulator = Accumulator(num_reflections);

    for (size_t j = 0; j < n_images; j++) {
        if (!reader.is_image_available(j)) {
            continue;
        }
        auto image = reader.get_image(j);
        logger.info("Processing image {}", j + 1);
        for (size_t k = 0; k < reflections_by_image[j].size(); k++) {
            size_t refl_id = reflections_by_image[j][k];
            if (constants.dont_integrate[refl_id]) {
                continue;
            }
            const auto &bbox = constants.computed_bounding_boxes[refl_id];
            if (constants.fg_algorithm == "ellipsoid") {
                // calculate dxy array (arrays of distances in kabsch space from pixel corners to xyzcal)
                int n_x = bbox.x_max - bbox.x_min + 1;
                int n_y = bbox.y_max - bbox.y_min + 1;
                std::vector<double> dxy_start(n_x
                                              * n_y);    // start of image (in rotation)
                std::vector<double> dxy_end(n_x * n_y);  // end of image (in rotation)
                double phidash_start = (constants.phi0 + (j * constants.dphi)) * constants.DEG2RAD;
                double phidash_end = (constants.phi0 + ((j + 1) * constants.dphi)) * constants.DEG2RAD;
                double phi_c = constants.phi_column(refl_id, 2);
                CoordinateSystem cs = constants.coord_system_vector[refl_id];
                Vector3d s1_this = Eigen::Map<Vector3d>(&constants.s1_vectors(refl_id, 0));
                for (int x = 0; x < n_x; x++) {
                    for (int y = 0; y < n_y; y++) {
                        int index = x + (y * (n_x));
                        std::array<double, 2> px_mm =
                          constants.panel.px_to_mm(x + bbox.x_min, y + bbox.y_min);
                        Vector3d s1_ = constants.panel.get_lab_coord(px_mm[0], px_mm[1]);
                        s1_.normalize();
                        Vector3d s1dash = s1_ * constants.s0_length;
                        Vector3d epsilon_coords_start =
                          cs.coords_from_s1vector(s1dash, phidash_start);
                        Vector3d epsilon_coords_end =
                          cs.coords_from_s1vector(s1dash, phidash_end);
                        if ((phidash_start > phi_c) && (phi_c < phidash_end)) {
                            epsilon_coords_start[2] = 0.0;
                            epsilon_coords_end[2] = 0.0;
                        }

                        dxy_start[index] =
                          ((epsilon_coords_start[0] * epsilon_coords_start[0]
                            + epsilon_coords_start[1] * epsilon_coords_start[1])
                           * constants.delta_b_r2)
                          + ((epsilon_coords_start[2] * epsilon_coords_start[2])
                             * constants.delta_m_r2);
                        dxy_end[index] =
                          ((epsilon_coords_end[0] * epsilon_coords_end[0]
                            + epsilon_coords_end[1] * epsilon_coords_end[1])
                           * constants.delta_b_r2)
                          + ((epsilon_coords_end[2] * epsilon_coords_end[2])
                             * constants.delta_m_r2);
                    }
                }
                // Now loop through pixels in bbox, adding to foreground or background if unmasked and in image.
                for (int x = bbox.x_min; x < bbox.x_max; x++) {
                    for (int y = bbox.y_min; y < bbox.y_max; y++) {
                        // The bounding box may extend beyond the image - if so, this is ok if it is only background
                        // but not if it is foreground.
                        int x_index = x - bbox.x_min;
                        int y_index = y - bbox.y_min;
                        int dxyindex = x_index + (y_index * n_x);
                        double d1 = dxy_start[dxyindex];
                        double d2 = dxy_start[dxyindex + 1];
                        double d3 = dxy_start[dxyindex + n_x];
                        double d4 = dxy_start[dxyindex + n_x + 1];
                        double d5 = dxy_end[dxyindex];
                        double d6 = dxy_end[dxyindex + 1];
                        double d7 = dxy_end[dxyindex + n_x];
                        double d8 = dxy_end[dxyindex + n_x + 1];
                        double d =
                          std::min(std::min(std::min(d1, d2), std::min(d3, d4)),
                                   std::min(std::min(d5, d6), std::min(d7, d8)));
                        int index = x + (y * image_fast);
                        bool in_image =
                          x >= 0 && x < image_fast && y >= 0 && y < image_slow;
                        if (d <= 1.0) {
                            // Foreground
                            if (in_image) {
                                if (mask[index] == 0) {  // masked
                                    accumulator.success[refl_id] = false;
                                } else {
                                    auto data = image.data.data()[index];
                                    accumulator.intensity[refl_id] += data;
                                    accumulator.nfg[refl_id] += 1;
                                    Vector3d I_times_c = {data * (x + 0.5),
                                                          data * (y + 0.5),
                                                          data * (j + 0.5)};
                                    accumulator.intensity_times_coord[refl_id] +=
                                      I_times_c;
                                }
                            } else {
                                accumulator.success[refl_id] = false;
                            }
                        } else {  // d>1.0
                            // Background
                            if (in_image) {
                                if (mask[index] != 0) {
                                    // In image, not masked
                                    auto data = image.data.data()[index];
                                    accumulator.bg[refl_id].add(data);
                                    accumulator.nbg[refl_id] += 1;
                                }
                            }
                        }
                    }
                }
            } else {
                // dials algorithm - a fixed ellipse (2D) across all images in the bbox
                int n_x = bbox.x_max - bbox.x_min + 1;
                int n_y = bbox.y_max - bbox.y_min + 1;
                std::vector<double> dxy(n_x * n_y);
                double phi =
                  constants.phi0
                  + (j * constants.dphi)
                      * constants.DEG2RAD;  // Required for calls but not actually used as don't use e3.
                CoordinateSystem cs = constants.coord_system_vector[refl_id];
                Vector3d s1_this = Eigen::Map<Vector3d>(&constants.s1_vectors(refl_id, 0));
                for (int x = 0; x < n_x; x++) {
                    for (int y = 0; y < n_y; y++) {
                        int index = x + (y * (n_x));
                        std::array<double, 2> px_mm =
                          constants.panel.px_to_mm(x + bbox.x_min, y + bbox.y_min);
                        Vector3d s1_ = constants.panel.get_lab_coord(px_mm[0], px_mm[1]);
                        s1_.normalize();
                        Vector3d s1dash = s1_ * constants.s0_length;
                        Vector3d epsilon_coords = cs.coords_from_s1vector(s1dash, phi);

                        dxy[index] = ((epsilon_coords[0] * epsilon_coords[0]
                                       + epsilon_coords[1] * epsilon_coords[1])
                                      * constants.delta_b_r2);
                    }
                }
                // Now loop through pixels in bbox, adding to foreground or background if unmasked and in image.
                for (int x = bbox.x_min; x < bbox.x_max; x++) {
                    for (int y = bbox.y_min; y < bbox.y_max; y++) {
                        // The bounding box may extend beyond the image - if so, this is ok if it iis only background
                        // but not if it is foreground.
                        int x_index = x - bbox.x_min;
                        int y_index = y - bbox.y_min;
                        int dxyindex = x_index + (y_index * n_x);
                        double d1 = dxy[dxyindex];
                        double d2 = dxy[dxyindex + 1];
                        double d3 = dxy[dxyindex + n_x];
                        double d4 = dxy[dxyindex + n_x + 1];
                        double d = std::min(std::min(d1, d2), std::min(d3, d4));
                        int index = x + (y * image_fast);
                        bool in_image =
                          x >= 0 && x < image_fast && y >= 0 && y < image_slow;

                        if (d <= 1.0) {
                            // Foreground
                            if (in_image) {
                                if (mask[index] == 0) {  // masked)
                                    accumulator.success[refl_id] = false;
                                } else {
                                    auto data = image.data.data()[index];
                                    accumulator.intensity[refl_id] += data;
                                    accumulator.nfg[refl_id] += 1;
                                    Vector3d I_times_c = {data * (x + 0.5),
                                                          data * (y + 0.5),
                                                          data * (j + 0.5)};
                                    accumulator.intensity_times_coord[refl_id] +=
                                      I_times_c;
                                }
                            } else {
                                accumulator.success[refl_id] = false;
                            }
                        } else {  // d>1.0
                            // Background
                            if (in_image) {
                                if (mask[index] != 0) {
                                    // In image, not masked
                                    auto data = image.data.data()[index];
                                    accumulator.bg[refl_id].add(data);
                                    accumulator.nbg[refl_id] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }*/
#pragma endregion Loop over images and perform summation

#pragma region Finalize calculations
    // Convert bbox data to reflection table format
    std::vector<double> computed_bbox_data;
    computed_bbox_data.reserve(num_reflections * 6);
    for (const auto &bbox : computed_bounding_boxes) {
        computed_bbox_data.insert(computed_bbox_data.end(),
                                  {bbox.x_min,
                                   bbox.x_max,
                                   bbox.y_min,
                                   bbox.y_max,
                                   static_cast<double>(bbox.z_min),
                                   static_cast<double>(bbox.z_max)});
    }
    // FIXME - these are not yet output as seems to cause an issue with dials?
    // Add computed bounding boxes to reflection table
    //integrated_data.add_column(
    //  "baseline_bbox", std::vector<size_t>{num_reflections, 6}, computed_bbox_data);

    // Do the calculations to determine the background-subtracted intensities as well as other quantities of interest.
    std::vector<double> intensities(num_reflections);
    std::vector<double> variances(num_reflections);
    std::vector<double> partialities(num_reflections);
    std::vector<double> lp_corrections(num_reflections);
    std::vector<double> xyzobs_px(num_reflections * 3);
    std::vector<double> d_values(num_reflections);
    std::vector<double> mean_bg(num_reflections);
    std::vector<double> bg_sum_value(num_reflections);
    // FIXME need to include robust outlier rejection in background calculation (glm, constant3dmodel)
    for (int i = 0; i < num_reflections; i++) {
        if (nbg_accumulators[i]) {
            auto [bg, total_bg_counts] =
              compute_background_constant_3d(bg_accumulators[i]);
            mean_bg[i] = bg;
            bg_sum_value[i] = total_bg_counts;
            double I = intensity_accumulators[i] - bg;
            intensities[i] = I;
            double m_n = nfg_accumulators[i] / nbg_accumulators[i];
            variances[i] = std::abs(I) + (std::abs(bg) * (1.0 + m_n));
            if ((nfg_accumulators[i] > 0) && (intensity_accumulators[i] > 0)) {
                Vector3d com =
                  intensity_times_coord_accumulators[i] / intensity_accumulators[i];
                xyzobs_px[i * 3] = com[0];
                xyzobs_px[i * 3 + 1] = com[1];
                xyzobs_px[i * 3 + 2] = com[2];
            } else {  // zero intensity, so com is just centre of box (dials convention from centroid/simple/algorithm.h).
                xyzobs_px[i * 3] = 0.5
                                   * (computed_bounding_boxes[i].x_min
                                      + computed_bounding_boxes[i].x_max);
                xyzobs_px[i * 3 + 1] = 0.5
                                       * (computed_bounding_boxes[i].y_min
                                          + computed_bounding_boxes[i].y_max);
                xyzobs_px[i * 3 + 2] = 0.5
                                       * (computed_bounding_boxes[i].z_min
                                          + computed_bounding_boxes[i].z_max);
            }
        } else {
            success[i] = false;
        }
    }
    // Calculate the partiality, LP, d of all reflections
    Vector3d pn = beam.get_polarization_normal();
    double pf = beam.get_polarization_fraction();
    LPCorrection lpcalculator(s0, pn, pf, rotation_axis);
    for (int i = 0; i < num_reflections; i++) {
        double xyzcal_px_z = phi_column(i, 2) * RAD2DEG / osc_width;
        double phi = osc_start + ((xyzcal_px_z + 1 - image_range_start) * osc_width);
        double phia =
          osc_start
          + ((computed_bounding_boxes[i].z_min + 1 - image_range_start) * osc_width);
        double phib =
          osc_start
          + ((computed_bounding_boxes[i].z_max + 1 - image_range_start) * osc_width);
        double zeta = coord_system_vector[i].zeta();
        // Partiality calculation
        double c = std::abs(zeta) / (std::sqrt(2.0) * sigma_m);
        double p = 0.5 * (std::erf(c * (phib - phi)) - std::erf(c * (phia - phi)));
        partialities[i] = p;
        // LP calculation
        Vector3d s1_this = Eigen::Map<Vector3d>(&s1_vectors(i, 0));
        lp_corrections[i] = lpcalculator.calculate(s1_this);
        // resolution calculation
        std::array<int, 3> hkl_this = {
          hkl_vectors[i * 3], hkl_vectors[i * 3 + 1], hkl_vectors[i * 3 + 2]};
        d_values[i] = cell.calculate_d(hkl_this);
    }
#pragma endregion Finalize calculations

#pragma region Make output table and save
    // Add data to reflection table.
    integrated_data.add_column("intensity.sum.value", num_reflections, 1, intensities);
    integrated_data.add_column("intensity.sum.variance", num_reflections, 1, variances);
    integrated_data.add_column("partiality", num_reflections, 1, partialities);
    integrated_data.add_column("miller_index", num_reflections, 3, hkl_vectors);
    integrated_data.add_column("lp", num_reflections, 1, lp_corrections);
    integrated_data.add_column("d", num_reflections, 1, d_values);
    std::vector<double> xyzcal_mm = std::vector<double>(
      phi_column.data_handle(), phi_column.data_handle() + phi_column.size());
    std::vector<double> s1 = std::vector<double>(
      s1_vectors.data_handle(), s1_vectors.data_handle() + s1_vectors.size());
    std::vector<int> id = std::vector<int>(num_reflections, 0);
    integrated_data.add_column("xyzcal.mm", num_reflections, 3, xyzcal_mm);
    integrated_data.add_column("xyzobs.px.value", num_reflections, 3, xyzobs_px);
    integrated_data.add_column("s1", num_reflections, 3, s1);
    integrated_data.add_column("id", num_reflections, 1, id);
    integrated_data.add_column(
      "num_pixels.background",
      num_reflections,
      1,
      nbg_accumulators);  // Useful for debug but not required for downstream
    integrated_data.add_column(
      "num_pixels.foreground",
      num_reflections,
      1,
      nfg_accumulators);  // Useful for debug but not required for downstream
    integrated_data.add_column(
      "background.sum.value",
      num_reflections,
      1,
      bg_sum_value);  // Useful for debug but not required for downstream
    integrated_data.add_column(
      "background.mean",
      num_reflections,
      1,
      mean_bg);  // Useful for debug but not required for downstream
    std::vector<std::size_t> final_flags(num_reflections, IntegratedSum);
    integrated_data.add_column(std::string("flags"), num_reflections, 1, final_flags);

    // Only output successfully integrated reflections.
    /*for (int i=0;i<accumulator.success.size();i++){
        success[i] = success[i] && accumulator.success[i];
    }*/

    ReflectionTable success_data = integrated_data.select(success);
    int n_integrated =
      success_data.column<double>("intensity.sum.value").value().extent(0);
    // Other things to potentially output xyzcal.px, xyzobs.mm.value, xyzobs.mm.variance, xyzobs.px.variance?

    logger.info("Saving results to: {}", output_file);
    success_data.write(output_file);
    logger.info("Baseline integration complete!");
    logger.info("Summary:");
    logger.info("  {} reflections processed", num_reflections);
    logger.info("  {} bounding boxes computed", computed_bounding_boxes.size());
    logger.info("  {} reflections successfully integrated", n_integrated);
    logger.info("Results saved to: {}", output_file);

#pragma endregion Make output table and save

    return 0;
}
#pragma endregion Application Entry
