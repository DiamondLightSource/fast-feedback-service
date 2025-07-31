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

// Define a 2D mdspan type alias for convenience
using mdspan_2d =
  std::experimental::mdspan<double, std::experimental::dextents<size_t, 2>>;

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

/**
 * @brief Structure to hold pixel coordinate extents for reflection bounding
 * boxes.
 *
 * Contains the minimum and maximum bounds in detector pixel coordinates (x, y)
 * and image numbers (z) that define the region of interest around each
 * reflection for integration.
 */
struct BoundingBoxExtents {
    double x_min, x_max;  ///< Detector x-pixel range (fast axis)
    double y_min, y_max;  ///< Detector y-pixel range (slow axis)
    int z_min, z_max;     ///< Image number range (rotation axis)
};

/**
 * @brief Compute bounding box extents for reflection integration using the
 * Kabsch coordinate system.
 *
 * 1. Calculates angular divergence parameters Δb and Δm
 * 2. Projects these divergences onto the Kabsch coordinate system to find
 *    the corners of the integration region in reciprocal space
 * 3. Transforms these reciprocal space coordinates back to detector pixel
 *    coordinates and image numbers to define practical bounding boxes
 *
 * The method accounts for the non-orthonormal nature of the Kabsch basis
 * and ensures that the bounding boxes encompass the full extent of each
 * reflection's diffraction profile.
 *
 * @param s0 Incident beam vector (s₀), units of 1/Å
 * @param rot_axis Unit goniometer rotation axis vector (m₂)
 * @param s1_vectors Matrix of predicted s₁ vectors for all reflections,
 *                   shape (num_reflections, 3)
 * @param phi_positions Matrix containing reflection positions, where the third
 *                      column contains φᶜ values in radians
 * @param num_reflections Number of reflections to process
 * @param sigma_b Beam divergence standard deviation (σb), in reciprocal space
 *                units
 * @param sigma_m Mosaicity standard deviation (σm), in reciprocal space units
 * @param panel Detector panel object for coordinate transformations
 * @param scan Scan object containing oscillation and image range information
 * @param beam Beam object for wavelength and other beam properties
 * @param n_sigma Number of standard deviations to include in the bounding box
 *                (default: 3.0)
 * @param sigma_b_multiplier Additional multiplier for beam divergence
 *                           (default: 2.0, called 'm' in DIALS)
 * @return Vector of BoundingBoxExtents structures, one per reflection
 */
std::vector<BoundingBoxExtents> compute_kabsch_bounding_boxes(
  const Eigen::Vector3d& s0,
  const Eigen::Vector3d& rot_axis,
  const mdspan_2d& s1_vectors,
  const mdspan_2d& phi_positions,
  const size_t num_reflections,
  const double sigma_b,                     // σb from arguments
  const double sigma_m,                     // σm from arguments
  const Panel& panel,                       // Panel for coordinate transformations
  const Scan& scan,                         // Scan for oscillation and image range data
  const MonochromaticBeam& beam,            // Beam for wavelength
  const double n_sigma = 3.0,               // Number of standard deviations
  const double sigma_b_multiplier = 2.0) {  // m parameter from DIALS

    std::vector<BoundingBoxExtents> extents;
    extents.reserve(num_reflections);

    /*
    * Tolerance for detecting when a reflection is nearly parallel to
    * the rotation axis. When ζ = m₂ · e₁ approaches zero, it indicates
    * the reflection's scattering plane is nearly parallel to the
    * goniometer rotation axis, making the φ-to-image conversion
    * numerically unstable. This threshold (1e-10) is chosen based on
    * geometric considerations rather than pure floating-point precision
    * - it represents a practical limit for "nearly parallel" geometry
    * where the standard bounding box calculation should be bypassed in
    * favor of spanning the entire image range.
    */
    static constexpr double ZETA_TOLERANCE = 1e-10;

    // Calculate the angular divergence parameters:
    // Δb = nσ × σb × m (beam divergence extent)
    // Δm = nσ × σm (mosaicity extent)
    double delta_b = n_sigma * sigma_b * sigma_b_multiplier;
    double delta_m = n_sigma * sigma_m;

    // Extract experimental parameters needed for coordinate transformations
    const auto [osc_start, osc_width] = scan.get_oscillation();
    int image_range_start = scan.get_image_range()[0];
    int image_range_end = scan.get_image_range()[1];
    double wl = beam.get_wavelength();
    Matrix3d d_matrix_inv = panel.get_d_matrix().inverse();

    // Process each reflection individually
    for (size_t i = 0; i < num_reflections; ++i) {
        // Extract reflection centroid data
        Eigen::Vector3d s1_c(
          s1_vectors(i, 0), s1_vectors(i, 1), s1_vectors(i, 2));  // s₁ᶜ from s1_vectors
        double phi_c = (phi_positions(i, 2));  // φᶜ from xyzcal.mm column

        // Construct the Kabsch coordinate system for this reflection
        // e1 = s₁ᶜ × s₀ / |s₁ᶜ × s₀| (perpendicular to scattering plane)
        Eigen::Vector3d e1 = s1_c.cross(s0).normalized();
        // e2 = s₁ᶜ × e₁ / |s₁ᶜ × e₁| (within scattering plane, orthogonal to e1)
        Eigen::Vector3d e2 = s1_c.cross(e1).normalized();

        double s1_len = s1_c.norm();

        // Calculate s′ vectors at the four corners of the integration region
        // These correspond to the extremes: (±Δb, ±Δb) in Kabsch coordinates
        std::vector<Eigen::Vector3d> s_prime_vectors;
        static constexpr std::array<std::pair<int, int>, 4> corner_signs = {
          {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};

        for (auto [e1_sign, e2_sign] : corner_signs) {
            // Project Δb divergences onto Kabsch basis vectors
            // p represents the displacement in reciprocal space
            Eigen::Vector3d p =
              (e1_sign * delta_b * e1 / s1_len) + (e2_sign * delta_b * e2 / s1_len);

            // Debug output for the Ewald sphere calculation
            double p_magnitude = p.norm();
            logger.trace(
              "Reflection {}, corner ({},{}): p.norm()={:.6f}, s1_len={:.6f}, "
              "delta_b={:.6f}",
              i,
              e1_sign,
              e2_sign,
              p_magnitude,
              s1_len,
              delta_b);

            // Ensure the resulting s′ vector lies on the Ewald sphere
            // This involves solving: |s′|² = |s₁ᶜ|² for the correct magnitude
            double b = s1_len * s1_len - p.dot(p);
            if (b < 0) {
                logger.error(
                  "Negative b value: {:.6f} for reflection {} (p.dot(p)={:.6f}, "
                  "s1_len²={:.6f})",
                  b,
                  i,
                  p.dot(p),
                  s1_len * s1_len);
                logger.error(
                  "This means the displacement vector is too large for the Ewald "
                  "sphere");
                // Skip this corner or use a fallback approach
                continue;
            }
            double d = -(p.dot(s1_c) / s1_len) + std::sqrt(b);

            logger.trace("Reflection {}: b={:.6f}, d={:.6f}", i, b, d);

            // Construct the s′ vector: s′ = (d × ŝ₁ᶜ) + p
            Eigen::Vector3d s_prime = (d * s1_c / s1_len) + p;
            s_prime_vectors.push_back(s_prime);
        }

        // Transform s′ vectors back to detector coordinates using Panel's get_ray_intersection
        std::vector<std::pair<double, double>> detector_coords;
        for (const auto& s_prime : s_prime_vectors) {
            // Direct conversion from s′ vector to detector coordinates
            // get_ray_intersection returns coordinates in mm
            std::array<double, 2> xy_mm = panel.get_ray_intersection(s_prime);

            // Convert from mm to pixels using the new mm_to_px function
            std::array<double, 2> xy_pixels = panel.mm_to_px(xy_mm[0], xy_mm[1]);

            detector_coords.push_back({xy_pixels[0], xy_pixels[1]});
        }

        // Determine the bounding box in detector coordinates
        // Find minimum and maximum coordinates from the four corners
        auto [min_x_it, max_x_it] = std::minmax_element(
          detector_coords.begin(),
          detector_coords.end(),
          [](const auto& a, const auto& b) { return a.first < b.first; });
        auto [min_y_it, max_y_it] = std::minmax_element(
          detector_coords.begin(),
          detector_coords.end(),
          [](const auto& a, const auto& b) { return a.second < b.second; });

        BoundingBoxExtents bbox;
        // Use floor/ceil as specified in the paper: xmin = floor(min([x1,x2,x3,x4]))
        bbox.x_min = std::floor(min_x_it->first);
        bbox.x_max = std::ceil(max_x_it->first);
        bbox.y_min = std::floor(min_y_it->second);
        bbox.y_max = std::ceil(max_y_it->second);

        // Calculate the image range (z-direction) using mosaicity parameter Δm
        // The extent in φ depends on the geometry factor ζ = m₂ · e₁
        double zeta = rot_axis.dot(e1);
        if (std::abs(zeta) > ZETA_TOLERANCE) {  // Avoid division by zero
            // Convert angular extents to rotation angles: φ′ = φᶜ ± Δm/ζ
            double phi_plus = phi_c + delta_m / zeta;
            double phi_minus = phi_c - delta_m / zeta;

            // Convert phi angles from radians to degrees before using scan parameters
            double phi_plus_deg = phi_plus * 180.0 / M_PI;
            double phi_minus_deg = phi_minus * 180.0 / M_PI;

            // Transform rotation angles to image numbers using scan parameters
            double z_plus =
              image_range_start - 1 + ((phi_plus_deg - osc_start) / osc_width);
            double z_minus =
              image_range_start - 1 + ((phi_minus_deg - osc_start) / osc_width);

            // Clamp to the actual image range and use floor/ceil for integer bounds
            bbox.z_min =
              std::max(image_range_start, (int)std::floor(std::min(z_plus, z_minus)));
            bbox.z_max =
              std::min(image_range_end, (int)std::ceil(std::max(z_plus, z_minus)));
        } else {
            // Handle degenerate case where reflection is parallel to rotation axis
            // In this case, the reflection spans the entire image range
            bbox.z_min = image_range_start;
            bbox.z_max = image_range_end;
        }

        extents.push_back(bbox);
    }

    return extents;
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

    // Load reflection data
    logger.info("Loading data from file: {}", reflection_file);
    ReflectionTable reflections(reflection_file);

    // Display column names
    std::string column_names_str;
    for (const auto& name : reflections.get_column_names()) {
        column_names_str += "\n\t- " + name;
    }
    logger.info("Column names: {}", column_names_str);

    // Extract required columns and dereference optionals
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

    auto bbox_column_opt = reflections.column<int>("bbox");
    if (!bbox_column_opt) {
        logger.error("Column 'bbox' not found in reflection data.");
        return 1;
    }
    auto bbox_column = *bbox_column_opt;

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
    size_t num_reflections = s1_vectors.extent(0);
    const auto [osc_start, osc_width] = scan.get_oscillation();
    int image_range_start = scan.get_image_range()[0];
    double wl = beam.get_wavelength();

    // Compute new bounding boxes using Kabsch coordinate system

    logger.info("Computing new Kabsch bounding boxes for {} reflections",
                num_reflections);

    // Compute new bounding boxes
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

    // Convert to reflection table format for comparison
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

    // Compare with existing bounding boxes
    logger.info("Comparing computed bounding boxes with existing bbox column");
    logger.trace("First 5 bounding box comparisons:");
    for (size_t i = 0; i < std::min<size_t>(5, num_reflections); ++i) {
        // Existing bbox
        double ex_x_min = bbox_column(i, 0);
        double ex_x_max = bbox_column(i, 1);
        double ex_y_min = bbox_column(i, 2);
        double ex_y_max = bbox_column(i, 3);
        int ex_z_min = static_cast<int>(bbox_column(i, 4));
        int ex_z_max = static_cast<int>(bbox_column(i, 5));

        // Computed bbox
        const auto& comp_bbox = computed_bounding_boxes[i];

        logger.trace("bbox[{}]: existing x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] z=[{},{}]",
                     i,
                     ex_x_min,
                     ex_x_max,
                     ex_y_min,
                     ex_y_max,
                     ex_z_min,
                     ex_z_max);
        logger.trace("bbox[{}]: computed x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] z=[{},{}]",
                     i,
                     comp_bbox.x_min,
                     comp_bbox.x_max,
                     comp_bbox.y_min,
                     comp_bbox.y_max,
                     comp_bbox.z_min,
                     comp_bbox.z_max);
    }

    // Compute Kabsch coordinates for all voxels using existing bbox

    logger.info(
      "Computing Kabsch coordinates for voxel centers within existing bounding boxes");

    std::vector<double> voxel_kabsch_coords;  // ε₁, ε₂, ε₃ for each voxel center
    std::vector<int> voxel_reflection_ids;    // Which reflection each voxel belongs to
    std::vector<double> voxel_positions;      // x, y, z positions for each voxel center
    std::vector<double> voxel_s1_lengths;     // |s₁| for each voxel center

    // Convert global vectors to CUDA format once
    fastvec::Vector3D s0_cuda = fastvec::make_vector3d(s0.x(), s0.y(), s0.z());
    fastvec::Vector3D rotation_axis_cuda =
      fastvec::make_vector3d(rotation_axis.x(), rotation_axis.y(), rotation_axis.z());

    // Process each reflection's existing bounding box
    for (size_t refl_id = 0; refl_id < num_reflections; ++refl_id) {
        // Extract bounding box from existing column (format: x_min, x_max, y_min, y_max, z_min, z_max)
        double x_min = bbox_column(refl_id, 0);
        double x_max = bbox_column(refl_id, 1);
        double y_min = bbox_column(refl_id, 2);
        double y_max = bbox_column(refl_id, 3);
        int z_min = static_cast<int>(bbox_column(refl_id, 4));
        int z_max = static_cast<int>(bbox_column(refl_id, 5));

        // Debug: Calculate expected voxel count
        int x_count = static_cast<int>(x_max) - static_cast<int>(x_min);
        int y_count = static_cast<int>(y_max) - static_cast<int>(y_min);
        int z_count = z_max - z_min;
        int expected_voxels = x_count * y_count * z_count;

        logger.trace(
          "Reflection {}: bbox x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] z=[{},{}]",
          refl_id,
          x_min,
          x_max,
          y_min,
          y_max,
          z_min,
          z_max);
        logger.trace("Expected voxels: {} x {} x {} = {}",
                     x_count,
                     y_count,
                     z_count,
                     expected_voxels);

        // Get reflection centroid data
        Eigen::Vector3d s1_c_eigen(
          s1_vectors(refl_id, 0), s1_vectors(refl_id, 1), s1_vectors(refl_id, 2));
        fastvec::Vector3D s1_c_cuda =
          fastvec::make_vector3d(s1_c_eigen.x(), s1_c_eigen.y(), s1_c_eigen.z());
        double phi_c = phi_column(refl_id, 2);

        logger.trace(
          "Processing reflection {} with existing bbox x=[{:.1f},{:.1f}] "
          "y=[{:.1f},{:.1f}] z=[{},{}]",
          refl_id,
          x_min,
          x_max,
          y_min,
          y_max,
          z_min,
          z_max);

        // Collect all voxel data for this reflection for batch processing
        std::vector<fastvec::Vector3D> batch_s_pixels;
        std::vector<fastvec::scalar_t> batch_phi_pixels;
        std::vector<std::tuple<double, double, double>> batch_voxel_coords;

        // Iterate through all voxel centers in the bounding box
        for (int z = z_min; z < z_max; ++z) {
            double phi_pixel =
              osc_start + (z - image_range_start + 1.5) * osc_width / 180.0 * M_PI;

            for (int y = static_cast<int>(y_min); y < static_cast<int>(y_max); ++y) {
                for (int x = static_cast<int>(x_min); x < static_cast<int>(x_max);
                     ++x) {
                    // Use voxel center coordinates
                    // double voxel_x = x + 0.5;
                    // double voxel_y = y + 0.5;
                    double voxel_z = 1;  //z + 0.5;

                    std::array<double, 2> xy_mm = panel.px_to_mm(
                      static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5);

                    double voxel_x = xy_mm[0];
                    double voxel_y = xy_mm[1];

                    // Convert detector coordinates to lab coordinate using panel geometry
                    Vector3d lab_coord =
                      panel.get_d_matrix() * Vector3d(voxel_x, voxel_y, 1.0);

                    // logger.trace(
                    //   "Voxel center ({}, {}, {}): lab_coord = ({:.6f}, {:.6f}, {:.6f})",
                    //   voxel_x,
                    //   voxel_y,
                    //   voxel_z,
                    //   lab_coord.x(),
                    //   lab_coord.y(),
                    //   lab_coord.z());

                    // Convert lab coordinate to reciprocal space vector
                    Eigen::Vector3d s_pixel_eigen = lab_coord.normalized() / wl;
                    fastvec::Vector3D s_pixel_cuda = fastvec::make_vector3d(
                      s_pixel_eigen.x(), s_pixel_eigen.y(), s_pixel_eigen.z());

                    // Store for batch processing
                    batch_s_pixels.push_back(s_pixel_cuda);
                    batch_phi_pixels.push_back(
                      static_cast<fastvec::scalar_t>(phi_pixel));
                    batch_voxel_coords.emplace_back(voxel_x, voxel_y, voxel_z);
                }
            }
        }

        // Process all voxels for this reflection with CUDA if we have any
        if (!batch_s_pixels.empty()) {
            size_t num_voxels = batch_s_pixels.size();

            // Output arrays
            std::vector<fastvec::Vector3D> batch_eps_results(num_voxels);
            std::vector<fastvec::scalar_t> batch_s1_len_results(num_voxels);

            // Call CUDA function for voxel processing
            compute_voxel_kabsch(batch_s_pixels.data(),
                                 batch_phi_pixels.data(),
                                 s1_c_cuda,
                                 static_cast<fastvec::scalar_t>(phi_c),
                                 s0_cuda,
                                 rotation_axis_cuda,
                                 batch_eps_results.data(),
                                 batch_s1_len_results.data(),
                                 num_voxels);

            // Store results
            for (size_t i = 0; i < num_voxels; ++i) {
                const auto& [voxel_x, voxel_y, voxel_z] = batch_voxel_coords[i];
                const auto& eps = batch_eps_results[i];

                voxel_kabsch_coords.insert(voxel_kabsch_coords.end(),
                                           {eps.x, eps.y, eps.z});
                voxel_reflection_ids.push_back(static_cast<int>(refl_id));
                voxel_positions.insert(voxel_positions.end(),
                                       {voxel_x, voxel_y, voxel_z});
                voxel_s1_lengths.push_back(
                  static_cast<double>(batch_s1_len_results[i]));
            }
        }
    }

    size_t actual_voxels = voxel_reflection_ids.size();
    logger.info("Processed {} voxel centers, computed {} Kabsch coordinates",
                actual_voxels,
                voxel_kabsch_coords.size() / 3);

    // Add debugging information
    logger.info(
      "Vector sizes: kabsch_coords={}, reflection_ids={}, positions={}, s1_lengths={}",
      voxel_kabsch_coords.size(),
      voxel_reflection_ids.size(),
      voxel_positions.size(),
      voxel_s1_lengths.size());

    logger.info(
      "Expected sizes: kabsch_coords={}, reflection_ids={}, positions={}, "
      "s1_lengths={}",
      actual_voxels * 3,
      actual_voxels,
      actual_voxels * 3,
      actual_voxels);

    // Save results

    // Add computed bounding boxes to reflection table for comparison
    reflections.add_column(
      "computed_bbox", std::vector<size_t>{num_reflections, 6}, computed_bbox_data);

    // Create voxel data table
    ReflectionTable voxel_table;
    voxel_table.add_column(
      "kabsch_coordinates", std::vector<size_t>{actual_voxels, 3}, voxel_kabsch_coords);
    voxel_table.add_column(
      "reflection_id",
      std::vector<size_t>{actual_voxels, 1},
      std::vector<double>(voxel_reflection_ids.begin(), voxel_reflection_ids.end()));
    voxel_table.add_column(
      "pixel_coordinates", std::vector<size_t>{actual_voxels, 3}, voxel_positions);
    voxel_table.add_column(
      "voxel_s1_length", std::vector<size_t>{actual_voxels, 1}, voxel_s1_lengths);

    // Write output files
    reflections.write("output_reflections.h5");
    voxel_table.write("voxel_kabsch_data.h5");

    logger.info("Results saved to output_reflections.h5 and voxel_kabsch_data.h5");

    return 0;
}
#pragma endregion Application Entry