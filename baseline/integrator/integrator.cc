/**
 * @file baseline_integrator.cc
 * @brief Main application for baseline CPU-only integration
 */

#include <Eigen/Dense>
#include <dx2/beam.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "arg_parser.hpp"
#include "common.hpp"
#include "extent.cc"
#include "ffs_logger.hpp"
#include "kabsch.cc"
#include "version.hpp"

using json = nlohmann::json;

// Define a 2D mdspan type alias for convenience
using mdspan_2d =
  std::experimental::mdspan<double, std::experimental::dextents<size_t, 2>>;

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
          .action([&](const std::string& value) { _reflection_filepath = value; });

        add_argument("experiment")
          .metavar("experiments.expt")
          .help("Input experiment list")
          .action([&](const std::string& value) { _experiment_filepath = value; });
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

        add_argument("--sigma_m")
          .help("Sigma_m: Standard deviation of the rotation axis in reciprocal space.")
          .metavar("σm")
          .scan<'f', float>();

        add_argument("--sigma_b")
          .help(
            "Sigma_b: Standard deviation of the beam direction in reciprocal space.")
          .metavar("σb")
          .scan<'f', float>();

        add_argument("output")
          .help("Output file path")
          .metavar("output.h5")
          .default_value<std::string>("output.h5");
    }
};
#pragma endregion Argument Parsing

#pragma region Application Entry
int main(int argc, char** argv) {
    logger.info("Baseline Integrator Version: {}", FFS_VERSION);

    // Parse arguments
    auto parser = BaselineIntegratorArgumentParser(FFS_VERSION);
    auto args = parser.parse_args(argc, argv);
    const auto reflection_file = parser.reflections();
    const auto experiment_file = parser.experiment();

    float sigma_m = parser.get<float>("sigma_m");
    float sigma_b = parser.get<float>("sigma_b");
    float timeout = parser.get<float>("timeout");
    std::string output_file = parser.get<std::string>("output");

    logger.info("Parameters: sigma_m={:.6f}, sigma_b={:.6f}, timeout={:.1f}, output={}",
                sigma_m,
                sigma_b,
                timeout,
                output_file);

    // Guard against missing files
    if (!std::filesystem::exists(reflection_file)) {
        logger.error("Reflection file not found: {}", reflection_file);
        return 1;
    }
    if (!std::filesystem::exists(experiment_file)) {
        logger.error("Experiment file not found: {}", experiment_file);
        return 1;
    }

    // Load reflection data
    logger.info("Loading reflection data from: {}", reflection_file);
    ReflectionTable reflections(reflection_file);

    // Display column names for debugging
    std::string column_names_str;
    for (const auto& name : reflections.get_column_names()) {
        column_names_str += "\n\t- " + name;
    }
    logger.info("Available columns: {}", column_names_str);

    // Extract required columns
    auto s1_vectors_opt = reflections.column<double>("s1");
    if (!s1_vectors_opt) {
        logger.error("Column 's1' not found in reflection data.");
        return 1;
    }
    auto s1_vectors = *s1_vectors_opt;

    auto phi_column_opt = reflections.column<double>("xyzcal.mm");
    if (!phi_column_opt) {
        logger.error("Column 'xyzcal.mm' not found for phi positions.");
        return 1;
    }
    auto phi_column = *phi_column_opt;

    size_t num_reflections = s1_vectors.extent(0);
    logger.info("Processing {} reflections", num_reflections);

    // Parse experiment list from JSON
    logger.info("Loading experiment data from: {}", experiment_file);
    std::ifstream f(experiment_file);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error& ex) {
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
    } catch (const std::invalid_argument& ex) {
        logger.error(
          "Failed to construct Experiment from '{}': {}", experiment_file, ex.what());
        return 1;
    }

    // Extract experimental components
    MonochromaticBeam beam = expt.beam();
    Goniometer gonio = expt.goniometer();
    const Panel& panel = expt.detector().panels()[0];  // Assuming single panel detector
    const Scan& scan = expt.scan();

    // Extract vectors and parameters
    Eigen::Vector3d s0 = beam.get_s0();
    Eigen::Vector3d rotation_axis = gonio.get_rotation_axis();
    const auto [osc_start, osc_width] = scan.get_oscillation();
    int image_range_start = scan.get_image_range()[0];
    int image_range_end = scan.get_image_range()[1];
    double wl = beam.get_wavelength();

    logger.info("Experimental parameters:");
    logger.info("  Beam s0: ({:.6f}, {:.6f}, {:.6f})", s0.x(), s0.y(), s0.z());
    logger.info("  Wavelength: {:.6f} Å", wl);
    logger.info("  Rotation axis: ({:.6f}, {:.6f}, {:.6f})",
                rotation_axis.x(),
                rotation_axis.y(),
                rotation_axis.z());
    logger.info("  Oscillation: start={:.3f}°, width={:.3f}°", osc_start, osc_width);
    logger.info("  Image range: {} to {}", image_range_start, image_range_end);

    // Compute bounding boxes using baseline CPU algorithms
    logger.info("Computing Kabsch bounding boxes using baseline CPU algorithms...");
    auto computed_bounding_boxes = compute_kabsch_bounding_boxes(s0,
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

    // Convert to reflection table format
    std::vector<double> computed_bbox_data;
    computed_bbox_data.reserve(num_reflections * 6);
    for (const auto& bbox : computed_bounding_boxes) {
        computed_bbox_data.insert(computed_bbox_data.end(),
                                  {bbox.x_min,
                                   bbox.x_max,
                                   bbox.y_min,
                                   bbox.y_max,
                                   static_cast<double>(bbox.z_min),
                                   static_cast<double>(bbox.z_max)});
    }

    // Display some sample results
    logger.info("Sample bounding box results (first 5 reflections):");
    for (size_t i = 0; i < std::min<size_t>(5, num_reflections); ++i) {
        const auto& bbox = computed_bounding_boxes[i];
        logger.info("  bbox[{}]: x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] z=[{},{}]",
                    i,
                    bbox.x_min,
                    bbox.x_max,
                    bbox.y_min,
                    bbox.y_max,
                    bbox.z_min,
                    bbox.z_max);
    }

    // Compute Kabsch coordinates for sample voxel centers within bounding boxes
    logger.info(
      "Computing Kabsch coordinates for voxel centers using baseline CPU "
      "algorithms...");

    std::vector<double> voxel_kabsch_coords;  // ε₁, ε₂, ε₃ for each voxel center
    std::vector<int> voxel_reflection_ids;    // Which reflection each voxel belongs to
    std::vector<double> voxel_positions;      // x, y, z positions for each voxel center
    std::vector<double> voxel_s1_lengths;     // |s₁| for each voxel center

    // Process a subset of reflections for demonstration (to avoid excessive computation)
    size_t max_reflections_to_process = std::min<size_t>(10, num_reflections);
    logger.info("Processing voxels for first {} reflections as demonstration",
                max_reflections_to_process);

    for (size_t refl_id = 0; refl_id < max_reflections_to_process; ++refl_id) {
        const auto& bbox = computed_bounding_boxes[refl_id];

        // Get reflection centroid data
        Eigen::Vector3d s1_c(
          s1_vectors(refl_id, 0), s1_vectors(refl_id, 1), s1_vectors(refl_id, 2));
        double phi_c = phi_column(refl_id, 2);

        // Process a sample of voxels in the bounding box (every 5th voxel to keep computation manageable)
        int step = 5;
        int voxel_count = 0;

        for (int z = bbox.z_min; z < bbox.z_max; z += step) {
            double phi_pixel =
              osc_start + (z - image_range_start + 1.5) * osc_width / 180.0 * M_PI;

            for (int y = static_cast<int>(bbox.y_min); y < static_cast<int>(bbox.y_max);
                 y += step) {
                for (int x = static_cast<int>(bbox.x_min);
                     x < static_cast<int>(bbox.x_max);
                     x += step) {
                    // Convert pixel coordinates to lab coordinates
                    std::array<double, 2> xy_mm = panel.px_to_mm(
                      static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5);

                    Vector3d lab_coord =
                      panel.get_d_matrix() * Vector3d(xy_mm[0], xy_mm[1], 1.0);

                    // Convert lab coordinate to reciprocal space vector
                    Eigen::Vector3d s_pixel = lab_coord.normalized() / wl;

                    // Use baseline CPU pixel_to_kabsch algorithm
                    double s1_len_out;
                    Eigen::Vector3d kabsch_coords = pixel_to_kabsch(
                      s0, s1_c, phi_c, s_pixel, phi_pixel, rotation_axis, s1_len_out);

                    // Store results
                    voxel_kabsch_coords.insert(
                      voxel_kabsch_coords.end(),
                      {kabsch_coords.x(), kabsch_coords.y(), kabsch_coords.z()});
                    voxel_reflection_ids.push_back(static_cast<int>(refl_id));
                    voxel_positions.insert(
                      voxel_positions.end(),
                      {xy_mm[0], xy_mm[1], static_cast<double>(z)});
                    voxel_s1_lengths.push_back(s1_len_out);

                    voxel_count++;
                }
            }
        }

        logger.debug("Processed {} voxels for reflection {}", voxel_count, refl_id);
    }

    size_t actual_voxels = voxel_reflection_ids.size();
    logger.info("Processed {} voxel centers total", actual_voxels);

    // Create results and save to HDF5
    logger.info("Saving results to: {}", output_file);

    // Add computed bounding boxes to reflection table
    reflections.add_column(
      "baseline_bbox", std::vector<size_t>{num_reflections, 6}, computed_bbox_data);

    // Create voxel data table
    ReflectionTable voxel_table;
    if (actual_voxels > 0) {
        voxel_table.add_column("kabsch_coordinates",
                               std::vector<size_t>{actual_voxels, 3},
                               voxel_kabsch_coords);
        voxel_table.add_column("reflection_id",
                               std::vector<size_t>{actual_voxels, 1},
                               std::vector<double>(voxel_reflection_ids.begin(),
                                                   voxel_reflection_ids.end()));
        voxel_table.add_column(
          "pixel_coordinates", std::vector<size_t>{actual_voxels, 3}, voxel_positions);
        voxel_table.add_column(
          "voxel_s1_length", std::vector<size_t>{actual_voxels, 1}, voxel_s1_lengths);
    }

    // Write output files
    reflections.write(output_file);
    if (actual_voxels > 0) {
        std::string voxel_output =
          output_file.substr(0, output_file.find_last_of('.')) + "_voxels.h5";
        voxel_table.write(voxel_output);
        logger.info("Voxel data saved to: {}", voxel_output);
    }

    logger.info("Baseline integration complete!");
    logger.info("Summary:");
    logger.info("  {} reflections processed", num_reflections);
    logger.info("  {} bounding boxes computed", computed_bounding_boxes.size());
    logger.info("  {} voxel centers processed", actual_voxels);
    logger.info("Results saved to: {}", output_file);

    return 0;
}
#pragma endregion Application Entry
