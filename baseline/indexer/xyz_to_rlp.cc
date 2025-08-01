#include <math.h>

#include <Eigen/Dense>
#include <dx2/beam.hpp>
#include <dx2/detector.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/scan.hpp>
#include <experimental/mdspan>
#include <tuple>

using Eigen::Matrix3d;
using Eigen::Vector3d;

constexpr double DEG2RAD = M_PI / 180.0;

template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;

struct xyz_to_rlp_results {
    std::vector<double> rlp_data;
    std::vector<double> s1_data;
    std::vector<double> xyzobs_mm_data;
    mdspan_type<double> rlp;
    mdspan_type<double> s1;
    mdspan_type<double> xyzobs_mm;

    xyz_to_rlp_results(int extent)
        : rlp_data(extent * 3),
          s1_data(extent * 3),
          xyzobs_mm_data(extent * 3),
          rlp(rlp_data.data(), extent, 3),
          s1(s1_data.data(), extent, 3),
          xyzobs_mm(xyzobs_mm_data.data(), extent, 3) {}
};

/**
 * @brief Transform detector pixel coordinates into reciprocal space coordinates.
 * @param xyzobs_px A 1D array of detector pixel coordinates from a single panel.
 * @param panel A dx2 Panel object defining the corresponding detector panel.
 * @param beam A dx2 MonochromaticBeam object.
 * @param scan A dx2 Scan object.
 * @param gonio A dx2 Goniometer object.
 * @returns A struct containing reciprocal space coordinates, s1 vectors and pixel coordinates in mm.
 */
xyz_to_rlp_results xyz_to_rlp(const mdspan_type<double> &xyzobs_px,
                              const Panel &panel,
                              const MonochromaticBeam &beam,
                              const Scan &scan,
                              const Goniometer &gonio) {
    // Use the experimental models to perform a coordinate transformation from
    // pixel coordinates in detector space to reciprocal space, in units of
    // inverse angstrom.
    // An equivalent to dials flex_ext.map_centroids_to_reciprocal_space method
    xyz_to_rlp_results results(xyzobs_px.extent(0));  // extent of the underlying data.

    // Extract the quantities from the models that are needed for the calculation.
    Vector3d s0 = beam.get_s0();
    double wl = beam.get_wavelength();
    std::array<double, 2> oscillation = scan.get_oscillation();
    const auto [osc_start, osc_width] = scan.get_oscillation();
    int image_range_start = scan.get_image_range()[0];
    Matrix3d setting_rotation_inverse = gonio.get_setting_rotation().inverse();
    Matrix3d sample_rotation_inverse = gonio.get_sample_rotation().inverse();
    Vector3d rotation_axis = gonio.get_rotation_axis();
    Matrix3d d_matrix = panel.get_d_matrix();

    for (int i = 0; i < xyzobs_px.extent(0); ++i) {
        // first convert detector pixel positions into mm
        double x1 = xyzobs_px(i, 0);
        double x2 = xyzobs_px(i, 1);
        double x3 = xyzobs_px(i, 2);
        std::array<double, 2> xymm = panel.px_to_mm(x1, x2);
        // convert the image 'z' coordinate to rotation angle based on the scan data
        double rot_angle =
          (((x3 + 1 - image_range_start) * osc_width) + osc_start) * DEG2RAD;

        // calculate the s1 vector using the detector d matrix
        Vector3d m = {xymm[0], xymm[1], 1.0};
        Vector3d s1_i = d_matrix * m;
        s1_i.normalize();
        // convert into inverse ansgtroms
        Vector3d s1_this = s1_i / wl;
        results.s1(i, 0) = s1_this[0];
        results.s1(i, 1) = s1_this[1];
        results.s1(i, 2) = s1_this[2];
        results.xyzobs_mm(i, 0) = xymm[0];
        results.xyzobs_mm(i, 1) = xymm[1];
        results.xyzobs_mm(i, 2) = rot_angle;

        // now apply the goniometer matrices
        // see https://dials.github.io/documentation/conventions.html for full conventions
        // rlp = F^-1 * R'^-1 * S^-1 * (s1-s0)
        Vector3d S = setting_rotation_inverse * (s1_this - s0);
        double cos = std::cos(-1.0 * rot_angle);
        double sin = std::sin(-1.0 * rot_angle);
        // The DIALS equivalent to the code below is
        // rlp_this = S.rotate_around_origin(gonio.rotation_axis, -1.0 * rot_angle);
        Vector3d rlp_this = (S * cos)
                            + (rotation_axis * rotation_axis.dot(S) * (1 - cos))
                            + (sin * rotation_axis.cross(S));

        rlp_this = sample_rotation_inverse * rlp_this;
        results.rlp(i, 0) = rlp_this[0];
        results.rlp(i, 1) = rlp_this[1];
        results.rlp(i, 2) = rlp_this[2];
    }
    return results;  // Return the data and spans.
}

std::vector<double> ssx_xyz_to_rlp(
  const std::vector<double>& xyzobs_px,
  double wavelength,
  const Panel& panel
  ){
    Vector3d s0 = {0.0,0.0,-1.0/wavelength};
    std::vector<double> rlp(xyzobs_px.size(), 0);
    Matrix3d d_matrix = panel.get_d_matrix();
    double rot_angle = 0.0;
    for (int i = 0; i < xyzobs_px.size() / 3; ++i) {
        // first convert detector pixel positions into mm
        double x1 = xyzobs_px[i*3];
        double x2 = xyzobs_px[i*3 + 1];
        std::array<double, 2> xymm = panel.px_to_mm(x1, x2);

        // calculate the s1 vector using the detector d matrix
        Vector3d m = {xymm[0], xymm[1], 1.0};
        Vector3d s1_i = d_matrix * m;
        s1_i.normalize();
        // convert into inverse ansgtroms
        Vector3d s1_this = s1_i / wavelength;

        Vector3d S = s1_this - s0;
        rlp[i*3] = S[0];
        rlp[i*3+1] = S[1];
        rlp[i*3+2] = S[2];
    }
    return rlp;
}

Panel make_panel(
  double distance, double beam_center_x, double beam_center_y,
  double pixel_size_x, double pixel_size_y, int image_size_x, int image_size_y
  ){
  std::array<double, 2> beam_center = {beam_center_x, beam_center_y};
  std::array<double, 2> pixel_size = {pixel_size_x, pixel_size_y};
  std::array<int, 2> image_size = {image_size_x, image_size_y};
  Panel panel(distance, beam_center, pixel_size, image_size);
  panel.set_correction_parameters(0.45,3.9220781,true); //FIXME add to constructor
  return panel;
}

void px_to_mm(const mdspan_type<double> &px_input,
              mdspan_type<double> &mm_output,
              const Scan &scan,
              const Panel &panel) {
    const auto [osc_start, osc_width] = scan.get_oscillation();
    int image_range_start = scan.get_image_range()[0];
    for (int i = 0; i < px_input.extent(0); ++i) {
        std::array<double, 2> xymm = panel.px_to_mm(px_input(i, 0), px_input(i, 1));
        double rot_angle =
          (((px_input(i, 2) + 1 - image_range_start) * osc_width) + osc_start)
          * DEG2RAD;
        mm_output(i, 0) = xymm[0];
        mm_output(i, 1) = xymm[1];
        mm_output(i, 2) = rot_angle;
    }
}
