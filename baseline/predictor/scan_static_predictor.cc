#ifndef DIALS_STATIC_PREDICTOR
#define DIALS_STATIC_PREDICTOR
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <dx2/beam.hpp>
#include <dx2/detector.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>

constexpr double two_pi = 2 * M_PI;

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

constexpr size_t predicted_value = (1 << 0);  //predicted flag

inline double mod2pi(double angle) {
    // E.g. treat 359.9999999 as 360
    if (std::abs(angle - two_pi) <= 1e-7) {
        angle = two_pi;
    }
    return angle - two_pi * floor(angle / two_pi);
}

Vector3d unit_rotate_around_origin(Vector3d vec, Vector3d unit, double angle) {
    double cosang = std::cos(angle);
    Vector3d res = vec * cosang + (unit * (unit.dot(vec)) * (1.0 - cosang))
                   + (unit.cross(vec) * std::sin(angle));
    return res;
}

/**
 * @brief Predict the positions of the reflections from the experiment models. Updates the table.
 * @param beam The beam model.
 * @param gonio The goniometer model.
 * @param UB The crystal UB matrix.
 * @param panel The panel from the detector model.
 * @param reflections The reflection table.
 */
void simple_reflection_predictor(const MonochromaticBeam beam,
                                 const Goniometer gonio,
                                 const Matrix3d UB,
                                 const Panel& panel,
                                 ReflectionTable& reflections) {
    // actually a repredictor as assumes all successful.
    auto flags_ = reflections.column<std::size_t>("flags");
    auto& flags = flags_.value();
    auto s1_ = reflections.column<double>("s1");
    auto& s1 = s1_.value();
    auto xyzobs_ = reflections.column<double>("xyzobs_mm");
    const auto& xyzobs_mm = xyzobs_.value();
    auto entering_ = reflections.column<ReflectionTable::BoolEnum>("entering");
    const auto& entering = entering_.value();
    auto hkl_ = reflections.column<int>("miller_index");
    const auto& hkl = hkl_.value();

    std::vector<double> xyzcal_mm_data(xyzobs_mm.size(), 0.0);
    mdspan_type<double> xyzcal_mm;
    //mdspan_type<double> xyzcal_mm(xyzcal_mm_data.data(), xyzcal_mm_data.size() / 3, 3);
    auto xyzcal_ = reflections.column<double>("xyzcal_mm");
    if (xyzcal_.has_value()){
        xyzcal_mm = xyzcal_.value();
    }
    else {
        reflections.add_column(
            "xyzcal_mm", xyzobs_mm.extent(0), 3, xyzcal_mm_data);
        auto xyzcal_ = reflections.column<double>("xyzcal_mm");
        xyzcal_mm = xyzcal_.value();
    }

    // these setup bits are the same for all refls.
    Vector3d s0 = beam.get_s0();
    Matrix3d F = gonio.get_sample_rotation();   //fixed rot
    Matrix3d S = gonio.get_setting_rotation();  //setting rot
    Vector3d R = gonio.get_rotation_axis();
    Vector3d s0_ = S.inverse() * s0;
    Matrix3d FUB = F * UB;
    Vector3d m2 = R / R.norm();
    Vector3d s0_m2_plane = s0.cross(S * R);
    s0_m2_plane.normalize();

    Vector3d m1 = m2.cross(s0_);
    m1.normalize();  //vary with s0
    Vector3d m3 = m1.cross(m2);
    m3.normalize();  //vary with s0
    double s0_d_m2 = s0_.dot(m2);
    double s0_d_m3 = s0_.dot(m3);

    // now call predict_rays with h and UB for a given refl
    for (int i = 0; i < hkl.extent(0); i++) {
        const Eigen::Map<Vector3i> h(&hkl(i, 0));
        const Vector3d hf{static_cast<double>(h[0]),
                          static_cast<double>(h[1]),
                          static_cast<double>(h[2])};
        ReflectionTable::BoolEnum entering_i = entering(i, 0);

        Vector3d pstar0 = FUB * hf;
        double pstar0_len_sq = pstar0.squaredNorm();
        if (pstar0_len_sq > 4 * s0_.squaredNorm()) {
            flags(i, 0) = flags(i, 0) & ~predicted_value;
            continue;
        }
        double pstar0_d_m1 = pstar0.dot(m1);
        double pstar0_d_m2 = pstar0.dot(m2);
        double pstar0_d_m3 = pstar0.dot(m3);
        double pstar_d_m3 =
          (-(0.5 * pstar0_len_sq) - (pstar0_d_m2 * s0_d_m2)) / s0_d_m3;
        double rho_sq = (pstar0_len_sq - (pstar0_d_m2 * pstar0_d_m2));
        double psq = pstar_d_m3 * pstar_d_m3;
        if (rho_sq < psq) {
            flags(i, 0) = flags(i, 0) & ~predicted_value;
            continue;
        }
        //DIALS_ASSERT(rho_sq >= sqr(pstar_d_m3));
        double pstar_d_m1 = sqrt(rho_sq - (psq));
        double p1 = pstar_d_m1 * pstar0_d_m1;
        double p2 = pstar_d_m3 * pstar0_d_m3;
        double p3 = pstar_d_m1 * pstar0_d_m3;
        double p4 = pstar_d_m3 * pstar0_d_m1;

        double cosphi1 = p1 + p2;
        double sinphi1 = p3 - p4;
        double a1 = atan2(sinphi1, cosphi1);
        // ASSERT must be in range? is_angle_in_range

        // check each angle
        Vector3d pstar = S * unit_rotate_around_origin(pstar0, m2, a1);
        Vector3d s1_this = s0_ + pstar;
        ReflectionTable::BoolEnum this_entering = s1_this.dot(s0_m2_plane) < 0.
                                                    ? ReflectionTable::BoolEnum::TRUE
                                                    : ReflectionTable::BoolEnum::FALSE;
        double angle;
        if (this_entering == entering_i) {
            // use this s1 and a1 (mod 2pi)
            angle = mod2pi(a1);
        } else {
            double cosphi2 = -p1 + p2;
            double sinphi2 = -p3 - p4;
            double a2 = atan2(sinphi2, cosphi2);
            pstar = S * unit_rotate_around_origin(pstar0, m2, a2);
            s1_this = s0_ + pstar;
            this_entering = s1_this.dot(s0_m2_plane) < 0.
                              ? ReflectionTable::BoolEnum::TRUE
                              : ReflectionTable::BoolEnum::FALSE;
            assert(this_entering == entering_i);
            angle = mod2pi(a2);
        }

        std::array<double, 2> mm = panel.get_ray_intersection(s1_this);
        // match full turns
        double phiobs = xyzobs_mm(i, 2);
        // first fmod positive
        double val = std::fmod(phiobs, two_pi);
        while (val < 0) val += two_pi;
        double resid = angle - val;
        // second fmod positive
        double val2 = std::fmod(resid + M_PI, two_pi);
        while (val2 < 0) val2 += two_pi;
        val2 -= M_PI;
        xyzcal_mm(i, 0) = mm[0];
        xyzcal_mm(i, 1) = mm[1];
        xyzcal_mm(i, 2) = phiobs + val2;
        s1(i, 0) = s1_this[0];
        s1(i, 1) = s1_this[1];
        s1(i, 2) = s1_this[2];
        flags(i, 0) = flags(i, 0) | predicted_value;
    }
}

#endif  // DIALS_STATIC_PREDICTOR