/**
 * @file extent.cc
 * @brief Extent and bounding box algorithms for baseline CPU implementation
 */

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <dx2/beam.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>

#include "ffs_logger.hpp"

// Define a 2D mdspan type alias for convenience
using mdspan_2d =
  std::experimental::mdspan<double, std::experimental::dextents<size_t, 2>>;

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
  const Eigen::Vector3d &s0,
  const Eigen::Vector3d &rot_axis,
  const mdspan_2d &s1_vectors,
  const mdspan_2d &phi_positions,
  const size_t num_reflections,
  const double sigma_b,                     // σb from arguments
  const double sigma_m,                     // σm from arguments
  const Panel &panel,                       // Panel for coordinate transformations
  const Scan &scan,                         // Scan for oscillation and image range data
  const MonochromaticBeam &beam,            // Beam for wavelength
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
        for (const auto &s_prime : s_prime_vectors) {
            // Direct conversion from s′ vector to detector coordinates
            // get_ray_intersection returns coordinates in mm
            auto xy_mm_opt = panel.get_ray_intersection(s_prime);
            if (!xy_mm_opt) {
                continue;  // Skip this corner if no intersection
            }
            std::array<double, 2> xy_mm = *xy_mm_opt;

            // Convert from mm to pixels using the new mm_to_px function
            std::array<double, 2> xy_pixels = panel.mm_to_px(xy_mm[0], xy_mm[1]);

            detector_coords.push_back({xy_pixels[0], xy_pixels[1]});
        }

        // Determine the bounding box in detector coordinates
        // Find minimum and maximum coordinates from the four corners
        auto [min_x_it, max_x_it] = std::minmax_element(
          detector_coords.begin(),
          detector_coords.end(),
          [](const auto &a, const auto &b) { return a.first < b.first; });
        auto [min_y_it, max_y_it] = std::minmax_element(
          detector_coords.begin(),
          detector_coords.end(),
          [](const auto &a, const auto &b) { return a.second < b.second; });

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
