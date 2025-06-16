/**
  * @file integrator.cc
 */

#include "integrator.cuh"

#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>
#include <iostream>
#include <string>

#include "common.hpp"
#include "cuda_arg_parser.hpp"
#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "version.hpp"

#pragma region Argument Parsing
class IntegratorArgumentParser : public CUDAArgumentParser {
  public:
    IntegratorArgumentParser(std::string version) : CUDAArgumentParser(version) {
        add_h5read_arguments();      // Override to use refl + expt
        add_integrator_arguments();  // Add integrator-specific args
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

        _activated_h5read = true;
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
    }
};
#pragma endregion Argument Parsing

#pragma region Application Entry
int main(int argc, char** argv) {
    logger.info("Version: {}", FFS_VERSION);

    // Parse arguments
    auto parser = IntegratorArgumentParser(FFS_VERSION);
    auto args = parser.parse_args(argc, argv);
    const auto reflection_file = parser.reflections();
    const auto experiment_file = parser.experiment();
    float wait_timeout = parser.get<float>("timeout");
    float sigma_m = parser.get<float>("sigma_m");
    float sigma_b = parser.get<float>("sigma_b");

    // Guard against missing files
    if (!std::filesystem::exists(reflection_file)) {
        logger.error("Reflection file not found: {}", reflection_file);
        return 1;
    }
    if (!std::filesystem::exists(experiment_file)) {
        logger.error("Experiment file not found: {}", experiment_file);
        return 1;
    }

    logger.info("Loading data from file: {}", reflection_file);
    ReflectionTable reflections(reflection_file);

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

    // Load experiment model data?

    return 0;
}
#pragma endregion Application Entry