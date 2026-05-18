/**
 * @file lp_correction.cc
 * @brief Lorentz and polarization correction calculations
 */

#include "integrator/lp_correction.hpp"

#include <cmath>

using Eigen::Vector3d;

double lorentz_correction(const Vector3d &s0, const Vector3d &m2, const Vector3d &s1) {
    double s1_length = s1.norm();
    double s0_length = s0.norm();
    return std::abs(s1.dot(m2.cross(s0))) / (s0_length * s1_length);
}

double polarization_correction(const Vector3d &s0,
                               const Vector3d &pn,
                               double pf,
                               const Vector3d &s1) {
    double s1_length = s1.norm();
    double s0_length = s0.norm();
    double P1 = ((pn.dot(s1)) / s1_length);
    double P2 = (1.0 - 2.0 * pf) * (1.0 - P1 * P1);
    double P3 = (s1.dot(s0) / (s1_length * s0_length));
    double P4 = pf * (1.0 + P3 * P3);
    double P = P2 + P4;
    return P;
}

LPCorrection::LPCorrection(Vector3d s0, Vector3d pn, double pf, Vector3d m2)
    : s0_(s0), pn_(pn), pf_(pf), m2_(m2) {}

double LPCorrection::calculate(const Vector3d s1) {
    double L = lorentz_correction(s0_, m2_, s1);
    double P = polarization_correction(s0_, pn_, pf_, s1);
    return L / P;
}
