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
#include "cuda_arg_parser.hpp"
#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "kabsch.cuh"
#include "math/vector3d.cuh"
#include "version.hpp"

#pragma region Algorithms

/**
 * @brief Transform a pixel from reciprocal space into the local Kabsch
 * coordinate frame.
 *
 * Given a predicted reflection centre and a pixel's position in
 * reciprocal space, this function calculates the local Kabsch
 * coordinates (ε₁, ε₂, ε₃), which represent displacements along a
 * non-orthonormal basis defined by the scattering geometry.
 *
 * This is used to determine whether a pixel falls within the profile of
 * a reflection in Kabsch space, which allows summation or profile
 * integration to proceed in a geometry-invariant coordinate frame.
 *
 * @param s0 Incident beam vector (s₀), units of 1/Å
 * @param s1_c Predicted diffracted vector at the reflection centre
 * (s₁ᶜ), units of 1/Å
 * @param phi_c Rotation angle at the reflection centre (φᶜ), in radians
 * @param s_pixel Diffracted vector at the current pixel (S′), units of
 * 1/Å
 * @param phi_pixel Rotation angle at the pixel (φ′), in radians
 * @param rot_axis Unit goniometer rotation axis vector (m₂)
 * @param s1_len_out Optional output for magnitude of s₁ᶜ (|s₁|)
 * @return Eigen::Vector3d The local coordinates (ε₁, ε₂, ε₃) in Kabsch
 * space
 */
Eigen::Vector3d pixel_to_kabsch(const Eigen::Vector3d& s0,
                                const Eigen::Vector3d& s1_c,
                                double phi_c,
                                const Eigen::Vector3d& s_pixel,
                                double phi_pixel,
                                const Eigen::Vector3d& rot_axis,
                                double& s1_len_out) {
    // Define the local Kabsch basis vectors:
    // e1 is perpendicular to the scattering plane
    Eigen::Vector3d e1 = s1_c.cross(s0).normalized();

    // e2 lies within the scattering plane, orthogonal to e1
    Eigen::Vector3d e2 = s1_c.cross(e1).normalized();

    // e3 bisects the angle between s0 and s1
    Eigen::Vector3d e3 = (s1_c + s0).normalized();

    // Compute the length of the predicted diffracted vector (|s₁|)
    double s1_len = s1_c.norm();
    s1_len_out = s1_len;

    // Rotation offset between the pixel and reflection centre
    double dphi = phi_pixel - phi_c;

    // Compute the predicted diffracted vector at φ′
    Eigen::Vector3d s1_phi_prime = s1_c + e3 * dphi;

    // Difference vector between pixel's s′ and the φ′-adjusted centroid
    Eigen::Vector3d deltaS = s_pixel - s1_phi_prime;

    // ε₁: displacement along e1, normalised by |s₁|
    double eps1 = e1.dot(deltaS) / s1_len;

    // ε₂: displacement along e2, with correction for non-orthogonality to e3
    double eps2 = e2.dot(deltaS) / s1_len - (e2.dot(e3) * dphi) / s1_len;

    // ε₃: displacement along rotation axis, scaled by ζ = m₂ · e₁
    double zeta = rot_axis.dot(e1);
    double eps3 = zeta * dphi;

    return {eps1, eps2, eps3};
}
#pragma endregion Algorithms

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

    // Display column names
    std::string column_names_str;
    for (const auto& name : reflections.get_column_names()) {
        column_names_str += "\n\t- " + name;
    }
    logger.info("Column names: {}", column_names_str);

    auto s1_vectors = reflections.column<double>("s1");
    if (!s1_vectors) {
        logger.error("Column 's1' not found in reflection data.");
        return 1;
    }

    // Load phi positions (used later for φ′ - φ)
    auto phi_column = reflections.column<double>("xyzcal.mm");
    if (!phi_column) {
        logger.error("Column 'xyzcal.mm' not found for phi positions.");
        return 1;
    }

    // Print first 10 s1 vectors for debugging
    logger.trace("First 10 s1 vectors:");
    for (size_t i = 0; i < std::min<size_t>(s1_vectors->extent(0), 10); ++i) {
        logger.trace(fmt::format("s1[{}]:\n\t({}, {}, {})",
                                 i,
                                 (*s1_vectors)(i, 0),
                                 (*s1_vectors)(i, 1),
                                 (*s1_vectors)(i, 2)));
    }

    // Parse experiment list from JSON
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

    // Construct Experiment object and extract beam
    Experiment<MonochromaticBeam> expt;
    try {
        expt = Experiment<MonochromaticBeam>(elist_json_obj);
    } catch (const std::invalid_argument& ex) {
        logger.error(
          "Failed to construct Experiment from '{}': {}", experiment_file, ex.what());
        return 1;
    }
    MonochromaticBeam beam = expt.beam();
    Goniometer gonio = expt.goniometer();

    Eigen::Vector3d s0_eigen = beam.get_s0();
    Eigen::Vector3d rotation_axis_eigen = gonio.get_rotation_axis();

    fastfb::Vector3D s0(s0_eigen.x(), s0_eigen.y(), s0_eigen.z());
    fastfb::Vector3D rotation_axis(
      rotation_axis_eigen.x(), rotation_axis_eigen.y(), rotation_axis_eigen.z());

    size_t num_reflections = s1_vectors->extent(0);

    // Convert s1 vectors to fastfb::Vector3D array
    std::vector<fastfb::Vector3D> h_s1(num_reflections);
    std::vector<double> h_phi(num_reflections);

    for (size_t i = 0; i < num_reflections; ++i) {
        h_s1[i] = fastfb::Vector3D(
          (*s1_vectors)(i, 0), (*s1_vectors)(i, 1), (*s1_vectors)(i, 2));
        h_phi[i] = (*phi_column)(i, 2);  // xyzcal.mm third component is φᶜ (rad)
    }

    // Prepare output arrays
    std::vector<fastfb::Vector3D> h_eps(num_reflections);
    std::vector<double> h_s1_len(num_reflections);

    // Call CUDA Kabsch function
    try {
        compute_kabsch(h_s1.data(),
                       h_phi.data(),
                       s0,
                       rotation_axis,
                       h_eps.data(),
                       h_s1_len.data(),
                       num_reflections);
    } catch (const std::runtime_error& ex) {
        logger.error("CUDA Kabsch computation failed: {}", ex.what());
        return 1;
    }

    // Convert results to flat vector for reflection table storage
    std::vector<double> eps_centre;
    eps_centre.reserve(num_reflections * 3);

    for (size_t i = 0; i < num_reflections; ++i) {
        eps_centre.insert(eps_centre.end(), {h_eps[i].x, h_eps[i].y, h_eps[i].z});
    }

    reflections.add_column(
      "eps_centre", std::vector<size_t>{s1_vectors->extent(0), 3}, eps_centre);

    logger.trace("Centroid ε-vectors (should be ~0):");
    for (std::size_t i = 0; i < std::min<std::size_t>(5, eps_centre.size() / 3); ++i) {
        logger.trace("  refl {:>3}  (ε1,ε2,ε3) = ({:+.4e},{:+.4e},{:+.4e})",
                     i,
                     eps_centre[i * 3 + 0],
                     eps_centre[i * 3 + 1],
                     eps_centre[i * 3 + 2]);
    }

    // Display debug output
    logger.trace("First 5 Kabsch coordinates:");
    for (size_t i = 0; i < std::min<size_t>(eps_centre.size() / 3, 5); ++i) {
        logger.trace("kabsch[{}]: ({:.5f}, {:.5f}, {:.5f})",
                     i,
                     eps_centre[i * 3 + 0],
                     eps_centre[i * 3 + 1],
                     eps_centre[i * 3 + 2]);
    }

    // Add as a column to the reflection table
    reflections.write("output_reflections.h5");

    return 0;
}
#pragma endregion Application Entry