/**
  * @file integrator.cc
 */

#include "integrator.cuh"

#include <Eigen/Dense>
#include <dx2/beam.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "common.hpp"
#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "version.hpp"

std::vector<double> compute_kabsch_coordinates(
  const std::experimental::mdspan<const double, std::experimental::dextents<size_t, 2>>&
    s1,
  const Eigen::Vector3d& s0,
  const Eigen::Vector3d& rotation_axis,
  double sigma_m,
  double sigma_b) {
    size_t n = s1.extent(0);
    std::vector<double> kabsch_coords(n * 3);

    Eigen::Vector3d m = rotation_axis.normalized();
    Eigen::Vector3d b = s0.normalized();
    Eigen::Vector3d q = m.cross(b).normalized();  // orthogonal direction

    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3d s1_i(s1(i, 0), s1(i, 1), s1(i, 2));
        Eigen::Vector3d ds = s1_i - s0;

        double q_component = q.dot(ds) / sigma_b;
        double b_component = b.dot(ds) / sigma_b;
        double m_component = m.dot(ds) / sigma_m;

        kabsch_coords[i * 3 + 0] = q_component;
        kabsch_coords[i * 3 + 1] = b_component;
        kabsch_coords[i * 3 + 2] = m_component;
    }

    return kabsch_coords;
}

#pragma region Argument Parsing
class IntegratorArgumentParser : public CUDAArgumentParser {
  public:
    IntegratorArgumentParser(std::string version) : CUDAArgumentParser(version) {
        add_h5read_arguments();      // Override to use refl + expt
        add_integrator_arguments();  // Add integrator-specific args
    }

    void add_h5read_arguments() override {
        add_argument("reflections")
          .metavar("strong.refl")
          .help("Input reflection table")
          .action([&](const std::string& value) { _arguments.reflection = value; });

        add_argument("experiment")
          .metavar("experiments.expt")
          .help("Input experiment list")
          .action([&](const std::string& value) { _arguments.experiment = value; });

        _activated_h5read = true;
    }

  private:
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
    }
};
#pragma endregion Argument Parsing

#pragma region Application Entry
int main(int argc, char** argv) {
    logger.info("Version: {}", FFS_VERSION);

    // Parse arguments
    auto parser = IntegratorArgumentParser(FFS_VERSION);
    auto args = parser.parse_args(argc, argv);
    float wait_timeout = parser.get<float>("timeout");
    float sigma_m = parser.get<float>("sigma_m");
    float sigma_b = parser.get<float>("sigma_b");

    // Guard against missing files
    if (!std::filesystem::exists(args.reflection)) {
        logger.error("Reflection file not found: {}", args.reflection);
        return 1;
    }
    if (!std::filesystem::exists(args.experiment)) {
        logger.error("Experiment file not found: {}", args.experiment);
        return 1;
    }

    logger.info("Loading data from file: {}", args.reflection);
    ReflectionTable reflections(args.reflection);

    auto column_names = reflections.get_column_names();
    std::string column_names_str;
    for (const auto& name : column_names) {
        column_names_str += "\n\t- " + name;
    }
    logger.info("Column names: {}", column_names_str);

    auto s1_vectors = reflections.column<double>("s1");
    if (!s1_vectors) {
        logger.error("Column 's1' not found in reflection data.");
        return 1;
    }

    // Print first 10 s1 vectors for debugging
    logger.trace("First 10 s1 vectors:");
    for (size_t i = 0; i < std::min<size_t>(s1_vectors->extent(0), 10); ++i) {
        logger.trace(fmt::format("s1[{}]:\n({}, {}, {})",
                                 i,
                                 (*s1_vectors)(i, 0),
                                 (*s1_vectors)(i, 1),
                                 (*s1_vectors)(i, 2)));
    }

    // Parse experiment list from JSON
    std::ifstream f(args.experiment);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error& ex) {
        logger.error("Failed to parse experiment file '{}': byte {}, {}",
                     args.experiment,
                     ex.byte,
                     ex.what());
        return 1;
    }

    // Construct Experiment object and extract beam
    Experiment<MonochromaticBeam> expt;
    try {
        expt = Experiment<MonochromaticBeam>(elist_json_obj);
    } catch (const std::invalid_argument& ex) {
        logger.error(
          "Failed to construct Experiment from '{}': {}", args.experiment, ex.what());
        return 1;
    }
    MonochromaticBeam beam = expt.beam();
    Goniometer gonio = expt.goniometer();

    Eigen::Vector3d s0 = beam.get_s0();
    Eigen::Vector3d rotation_axis = gonio.get_rotation_axis();

    // Compute Kabsch coordinates
    auto kabsch_coords =
      compute_kabsch_coordinates(*s1_vectors, s0, rotation_axis, sigma_m, sigma_b);

    // Debug: Print first few Kabsch vectors
    logger.trace("First 5 Kabsch coordinates:");
    for (size_t i = 0; i < std::min<size_t>(kabsch_coords.size() / 3, 5); ++i) {
        logger.trace(fmt::format("kabsch[{}]: ({:.5f}, {:.5f}, {:.5f})",
                                 i,
                                 kabsch_coords[i * 3 + 0],
                                 kabsch_coords[i * 3 + 1],
                                 kabsch_coords[i * 3 + 2]));
    }

    // reflections.add_column("kabsch", kabsch_coords.size() / 3, 3, kabsch_coords.data());

    return 0;
}
#pragma endregion Application Entry