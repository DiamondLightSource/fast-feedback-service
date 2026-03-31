#include <Eigen/Dense>

using Eigen::Vector3d;

double lorentz_correction(Vector3d const &s0, Vector3d const &m2, Vector3d const &s1) {
  double s1_length = s1.norm();
  double s0_length = s0.norm();
  return std::abs(s1.dot(m2.cross(s0))) / (s0_length * s1_length);
}
double polarization_correction(Vector3d const &s0,
                                 Vector3d const &pn,
                                 double pf,
                                 Vector3d const &s1) {
  double s1_length = s1.norm();
  double s0_length = s0.norm();
  double P1 = ((pn.dot(s1)) / s1_length);
  double P2 = (1.0 - 2.0 * pf) * (1.0 - P1 * P1);
  double P3 = (s1.dot(s0) / (s1_length * s0_length));
  double P4 = pf * (1.0 + P3 * P3);
  double P = P2 + P4;
  return P;
}

class LPCorrection {
  public:
    LPCorrection(Vector3d s0, Vector3d pn, double pf, Vector3d m2): s0_(s0), pn_(pn), pf_(pf), m2_(m2) {}
    double calculate(Vector3d const s1){
      double L = lorentz_correction(s0_, m2_, s1);
      double P = polarization_correction(s0_, pn_, pf_, s1);
      return L / P;
    }
  private:
    Vector3d s0_;
    Vector3d pn_;
    double pf_;
    Vector3d m2_;
    double s0_length;
};


class CoordinateSystem {
  public:
    CoordinateSystem(Vector3d m2, Vector3d s0, Vector3d s1, double phi) : s1_(s1), phi_(phi) {
      Vector3d m2_(m2);
      m2_.normalize();
      Vector3d e1_ = s1.cross(s0);
      e1_.normalize();
      Vector3d e2_ = s1.cross(e1_);
      e2_.normalize();
      double s1_length = s1.norm();
      scaled_e1_ = e1_ /  s1_length;
      scaled_e2_ = e2_ /  s1_length;
      zeta_ = m2_.dot(e1_);
    }
    Vector3d coords_from_s1vector(const Vector3d s_dash, double phi_dash){
      Vector3d coord = {scaled_e1_.dot(s_dash - s1_),
          scaled_e2_.dot(s_dash - s1_),
          zeta_ * (phi_dash - phi_)};
      return coord;
    }
    double zeta() const {
      return zeta_;
    }
  private:
    Vector3d s1_;
    double phi_;
    double zeta_;
    Vector3d scaled_e1_;
    Vector3d scaled_e2_;
};