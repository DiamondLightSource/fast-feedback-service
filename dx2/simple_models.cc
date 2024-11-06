#ifndef INDEXER_SIMPLE_MODELS
#define INDEXER_SIMPLE_MODELS
#include <Eigen/Dense>
#include <math.h>

using Eigen::Matrix3d;
using Eigen::Vector3d;

double attenuation_length(double mu, double t0,
                                   Vector3d s1,
                                   Vector3d fast,
                                   Vector3d slow,
                                   Vector3d origin) {
  Vector3d normal = fast.cross(slow);
  double distance = origin.dot(normal);
  if (distance < 0) {
    normal = -normal;
  }
  double cos_t = s1.dot(normal);
  //DXTBX_ASSERT(mu > 0 && cos_t > 0);
  return (1.0 / mu) - (t0 / cos_t + 1.0 / mu) * exp(-mu * t0 / cos_t);
}

class SimpleDetector {
public:
  Matrix3d d_matrix;
  double pixel_size;  // in mm
  double mu;
  double t0;
  bool parallax_correction;
  std::array<double, 2> px_to_mm(double x, double y) const;

  SimpleDetector(Matrix3d d_matrix, double pixel_size, double mu=0.0, double t0=0.0, bool parallax_correction=false) {
    this->d_matrix = d_matrix;
    this->pixel_size = pixel_size;
    this->mu = mu;
    this->t0 = t0;
    this->parallax_correction = parallax_correction;
  }
};

std::array<double, 2> SimpleDetector::px_to_mm(double x, double y) const{
  double x1 = x*pixel_size;
  double x2 = y*pixel_size;
  if (!parallax_correction){
    return std::array<double, 2>{x1,x2};
  }
  Vector3d fast = d_matrix.col(0);
  Vector3d slow = d_matrix.col(1);
  Vector3d origin = d_matrix.col(2);
  Vector3d s1 = origin + x1*fast + x2*slow;
  s1.normalize();
  double o = attenuation_length(mu, t0, s1, fast, slow, origin);
  double c1 = x1 - (s1.dot(fast))*o;
  double c2 = x2 - (s1.dot(slow))*o;
  return std::array<double, 2>{c1,c2};
}


#endif  // INDEXER_SIMPLE_MODELS