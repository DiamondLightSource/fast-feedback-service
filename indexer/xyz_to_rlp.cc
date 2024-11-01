#include <Eigen/Dense>
#include <chrono>
#include <math.h>

using Eigen::Matrix3d;
using Eigen::Vector3d;

class SimpleBeam {
public:
  double wavelength;
  Vector3d s0;

  SimpleBeam(double wavelength) {
    this->wavelength = wavelength;
    s0 = {0.0, 0.0, -1.0 / wavelength};
  }
};

class SimpleDetector {
public:
  Matrix3d d_matrix;
  double pixel_size;  // in mm

  SimpleDetector(Matrix3d d_matrix, double pixel_size) {
    this->d_matrix = d_matrix;
    this->pixel_size = pixel_size;
  }
};

class SimpleScan {
public:
  int image_range_start;
  double osc_start;
  double osc_width;

  SimpleScan(int image_range_start, double osc_start, double osc_width) {
    this->image_range_start = image_range_start;
    this->osc_start = osc_start;
    this->osc_width = osc_width;
  }
};

class SimpleGonio {
public:
  Matrix3d sample_rotation;
  Vector3d rotation_axis;
  Matrix3d setting_rotation;
  Matrix3d sample_rotation_inverse;
  Matrix3d setting_rotation_inverse;

  SimpleGonio(Matrix3d sample_rotation,
              Vector3d rotation_axis,
              Matrix3d setting_rotation) {
    this->sample_rotation = sample_rotation;
    rotation_axis.normalize();
    this->rotation_axis = rotation_axis;
    this->setting_rotation = setting_rotation;
    sample_rotation_inverse = this->sample_rotation.inverse();
    setting_rotation_inverse = this->setting_rotation.inverse();
  }
};

std::vector<Vector3d> xyz_to_rlp(
  std::vector<double> xyzobs_px,
  Matrix3d sample_rotation,
  Matrix3d detector_d_matrix,
  double wavelength,
  double pixel_size_mm,
  int image_range_start,
  double osc_start,
  double osc_width,
  Vector3d rotation_axis,
  Matrix3d setting_rotation) {
  auto start = std::chrono::system_clock::now();
  // An equivalent to dials flex_ext.map_centroids_to_reciprocal_space method
  SimpleBeam beam(wavelength);
  SimpleDetector detector(detector_d_matrix, pixel_size_mm);
  SimpleScan scan(image_range_start, osc_start, osc_width);
  SimpleGonio gonio(sample_rotation, rotation_axis, setting_rotation);

  float DEG2RAD = M_PI / 180.0;

  std::vector<Vector3d> rlp(xyzobs_px.size() / 3);
  for (int i = 0; i < rlp.size(); ++i) {
    // first convert detector pixel positions into mm
    int vec_idx= 3*i;
    double x1 = xyzobs_px[vec_idx];
    double x2 = xyzobs_px[vec_idx+1];
    double x3 = xyzobs_px[vec_idx+2];
    double x_mm = x1 * detector.pixel_size;
    double y_mm = x2 * detector.pixel_size;
    // convert the image 'z' coordinate to rotation angle based on the scan data
    double rot_angle =
      (((x3 + 1 - scan.image_range_start) * scan.osc_width) + scan.osc_start) * DEG2RAD;
    
    // calculate the s1 vector using the detector d matrix
    Vector3d m = {x_mm, y_mm, 1.0};
    Vector3d s1_i = detector.d_matrix * m;
    s1_i.normalize();
    // convert into inverse ansgtroms
    Vector3d s1_this = s1_i / beam.wavelength;
    
    // now apply the goniometer matrices
    // see https://dials.github.io/documentation/conventions.html for full conventions
    // rlp = F^-1 * R'^-1 * S^-1 * (s1-s0)
    Vector3d S = gonio.setting_rotation_inverse * (s1_this - beam.s0);
    double cos = std::cos(-1.0 * rot_angle);
    double sin = std::sin(-1.0 * rot_angle);
    Vector3d rlp_this = (S * cos)
                        + (gonio.rotation_axis * gonio.rotation_axis.dot(S) * (1 - cos))
                        + (sin * gonio.rotation_axis.cross(S));
    
    // lp_this = S.rotate_around_origin(gonio.rotation_axis, -1.0 * rot_angle);
    rlp[i] = gonio.sample_rotation_inverse * rlp_this;
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time for xyz_to_rlp: " << elapsed_seconds.count() << "s"
            << std::endl;
  return rlp;
}