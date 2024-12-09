#include <Eigen/Dense>
#include <math.h>
#include <dx2/detector.h>
#include <dx2/beam.h>
#include <dx2/scan.h>
#include <dx2/goniometer.h>

using Eigen::Matrix3d;
using Eigen::Vector3d;

std::vector<Vector3d> xyz_to_rlp(
  const std::vector<double> &xyzobs_px,
  const Panel &panel,
  const MonochromaticBeam &beam,
  const Scan &scan,
  const Goniometer &gonio) {
  // An equivalent to dials flex_ext.map_centroids_to_reciprocal_space method

  double DEG2RAD = M_PI / 180.0;
  std::vector<Vector3d> rlp(xyzobs_px.size() / 3);
  Vector3d s0 = beam.get_s0();
  double wl = beam.get_wavelength();
  std::array<double, 2> oscillation = scan.get_oscillation();
  double osc_width = oscillation[1];
  double osc_start = oscillation[0];
  int image_range_start = scan.get_image_range()[0];
  Matrix3d setting_rotation_inverse = gonio.get_setting_rotation().inverse();
  Matrix3d sample_rotation_inverse = gonio.get_sample_rotation().inverse();
  Vector3d rotation_axis = gonio.get_rotation_axis();
  Matrix3d d_matrix = panel.get_d_matrix();
  for (int i = 0; i < rlp.size(); ++i) {
    // first convert detector pixel positions into mm
    int vec_idx= 3*i;
    double x1 = xyzobs_px[vec_idx];
    double x2 = xyzobs_px[vec_idx+1];
    double x3 = xyzobs_px[vec_idx+2];
    std::array<double, 2> xymm = panel.px_to_mm(x1,x2);
    // convert the image 'z' coordinate to rotation angle based on the scan data
    double rot_angle =
      (((x3 + 1 - image_range_start) * osc_width) + osc_start) * DEG2RAD;
   
    // calculate the s1 vector using the detector d matrix
    Vector3d m = {xymm[0], xymm[1], 1.0};
    Vector3d s1_i = d_matrix * m;
    s1_i.normalize();
    // convert into inverse ansgtroms
    Vector3d s1_this = s1_i / wl;
    
    // now apply the goniometer matrices
    // see https://dials.github.io/documentation/conventions.html for full conventions
    // rlp = F^-1 * R'^-1 * S^-1 * (s1-s0)
    Vector3d S = setting_rotation_inverse * (s1_this - s0);
    double cos = std::cos(-1.0 * rot_angle);
    double sin = std::sin(-1.0 * rot_angle);
    // rlp_this = S.rotate_around_origin(gonio.rotation_axis, -1.0 * rot_angle);
    Vector3d rlp_this = (S * cos)
                        + (rotation_axis * rotation_axis.dot(S) * (1 - cos))
                        + (sin * rotation_axis.cross(S));
    
    
    rlp[i] = sample_rotation_inverse * rlp_this;
  }
  return rlp;
}