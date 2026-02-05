/**
 * @file extent.hpp
 * @brief Header for extent and bounding box algorithms
 */

#pragma once

#include <Eigen/Dense>
#include <dx2/beam.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <experimental/mdspan>
#include <vector>

// Define a 2D mdspan type alias for double precision
using mdspan_2d_double =
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
    // x and y may need to be a double for extent calculations
    // signed as some values are negative, but this should be clamped later
    int x_min, x_max;  ///< Detector x-pixel range (fast axis)
    int y_min, y_max;  ///< Detector y-pixel range (slow axis)
    int z_min, z_max;  ///< Image number range (rotation axis)
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
 * @param sigma_b Beam divergence standard deviation in RADIANS (σb)
 * @param sigma_m Mosaicity standard deviation in RADIANS (σm)
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
  const mdspan_2d_double &s1_vectors,
  const mdspan_2d_double &phi_positions,
  const size_t num_reflections,
  const double sigma_b,
  const double sigma_m,
  const Panel &panel,
  const Scan &scan,
  const MonochromaticBeam &beam,
  const double n_sigma = 3.0,
  const double sigma_b_multiplier = 2.0);
