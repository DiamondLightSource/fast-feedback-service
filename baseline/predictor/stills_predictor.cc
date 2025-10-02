#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <dx2/detector.hpp>
#include <dx2/reflection.hpp>

constexpr double two_pi = 2 * M_PI;

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

constexpr size_t predicted_value = (1 << 0);  //predicted flag


void simple_still_reflection_predictor(const Vector3d s0,
                                 const Matrix3d UB,
                                 const Panel &panel,
                                 ReflectionTable &reflections) {
    // actually a repredictor as assumes all successful.
    //auto flags_ = reflections.column<std::size_t>("flags");
    //auto &flags = flags_.value();
    auto hkl_ = reflections.column<int>("miller_index");
    const auto &hkl = hkl_.value();
    int n_refl = hkl.extent(0);
    std::vector<double> s1_data(3 * n_refl, 0.0);
    mdspan_type<double> s1;
    auto s1_ = reflections.column<double>("s1");
    if (s1_.has_value()){
        s1 = s1_.value();
    }
    else {
        reflections.add_column("s1", n_refl, 3, s1_data);
        auto s1_ = reflections.column<double>("s1");
        s1 = s1_.value();
    }
    std::vector<double> xyzcal_mm_data(3 * n_refl, 0.0);
    mdspan_type<double> xyzcal_mm;
    //mdspan_type<double> xyzcal_mm(xyzcal_mm_data.data(), xyzcal_mm_data.size() / 3, 3);
    auto xyzcal_ = reflections.column<double>("xyzcal.mm");
    if (xyzcal_.has_value()) {
        xyzcal_mm = xyzcal_.value();
    } else {
        reflections.add_column("xyzcal.mm", n_refl, 3, xyzcal_mm_data);
        auto xyzcal_ = reflections.column<double>("xyzcal.mm");
        xyzcal_mm = xyzcal_.value();
    }
    //std::vector<double> delpsical(flags.size(), 0.0);
    
    //Ray ray;
    //  ray = predict_ray_(h, ub);
    //  double delpsi = predict_ray_.get_delpsi();
    //  append_for_ray(p, h, ray, panel, delpsi);
    double s0_length = s0.norm();
    Vector3d unit_s0_ = s0 / s0_length;
    double lambda = 1. / s0_length;
    for (int i = 0; i < hkl.extent(0); i++) {
        const Eigen::Map<Vector3i> hkl_i(&hkl(i, 0));
        if (hkl_i[0] != 0 | hkl_i[1] != 0 | hkl_i[2] != 0){
            const Vector3d hf{static_cast<double>(hkl_i[0]),
                            static_cast<double>(hkl_i[1]),
                            static_cast<double>(hkl_i[2])};
            Vector3d q = UB * hf;
            Vector3d e1 = q.cross(unit_s0_);
            e1.normalize();
            Vector3d c0 = unit_s0_.cross(e1);
            c0.normalize();
            double qq = q.dot(q);
            double a = 0.5 * qq * lambda;
            double tmp = qq - (a * a);
            double b = std::sqrt(tmp);
            Vector3d r = -1.0 * a * unit_s0_ + b * c0;
            // Calculate delpsi value
            q.normalize();
            Vector3d q1 = q.cross(e1);
            q1.normalize();
            double delpsi_ = -1.0 * atan2(r.dot(q1), r.dot(q));

            // Calculate the Ray (default zero angle and 'entering' as false)
            Vector3d v = s0 + r;
            v.normalize();
            Vector3d s1_this = v * s0_length;
            //delpsical[i] = delpsi_;
            // now get ray intersection.
            std::array<double, 2> xymm  = panel.get_ray_intersection(s1_this);
            xyzcal_mm(i, 0) = xymm[0];
            xyzcal_mm(i, 1) = xymm[1];
            xyzcal_mm(i, 2) = delpsi_;
            s1(i, 0) = s1_this[0];
            s1(i, 1) = s1_this[1];
            s1(i, 2) = s1_this[2];
            //flags(i, 0) = flags(i, 0) | predicted_value;
        }
    }
}