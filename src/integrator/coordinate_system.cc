/**
 * @file coordinate_system.cc
 * @brief Kabsch reciprocal-space coordinate system for a reflection
 */

#include "integrator/coordinate_system.hpp"

using Eigen::Vector3d;

CoordinateSystem::CoordinateSystem(Vector3d m2, Vector3d s0, Vector3d s1, double phi)
    : s1_(s1), phi_(phi) {
    Vector3d m2_(m2);
    m2_.normalize();
    Vector3d e1_ = s1.cross(s0);
    e1_.normalize();
    Vector3d e2_ = s1.cross(e1_);
    e2_.normalize();
    double s1_length = s1.norm();
    scaled_e1_ = e1_ / s1_length;
    scaled_e2_ = e2_ / s1_length;
    zeta_ = m2_.dot(e1_);
}

Vector3d CoordinateSystem::coords_from_s1vector(const Vector3d s_dash,
                                                double phi_dash) const {
    Vector3d coord = {scaled_e1_.dot(s_dash - s1_),
                      scaled_e2_.dot(s_dash - s1_),
                      zeta_ * (phi_dash - phi_)};
    return coord;
}

double CoordinateSystem::zeta() const {
    return zeta_;
}
