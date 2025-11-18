#include <Eigen/Dense>
#include <cmath>
#include <nlohmann/json.hpp>

#include "utils.cc"
using json = nlohmann::json;

using Eigen::Matrix3d;
using Eigen::Vector3d;

#pragma once
/**
 * @brief Return a Ray object if a prediction is found within a give tolerance angle
 * 
 * @param index The Miller indices of the spot under consideration
 * @param A The (rotation * crystal setting matrix)
 * @param s0 The incident beam vector
 * @param dmin The minimum lattice spacing that can be resolved
 * @param delta_psi_tolerance The tolerance (in radians) of 
 * @return std::optional<Ray> 
 */
std::optional<Ray> predict_ray_monochromatic_stills(const std::array<int, 3> &index,
                                                    const Matrix3d &A,
                                                    const Vector3d &s0,
                                                    const double dmin,
                                                    const double delta_psi_tolerance) {
    const Vector3d hkl_vec{(double)index[0], (double)index[1], (double)index[2]};

    // Calculate the reciprocal space vectors, their magnitudes, and normalize them
    const Vector3d s0_unit = s0.normalized();
    const Vector3d r_vec = A * hkl_vec;
    const Vector3d r_unit = r_vec.normalized();
    const double s = s0.norm();
    const double r = r_vec.norm();

    // Find the angle by which the reciprocal lattice vector must be rotated to intersect
    // with the Ewald sphere, and the corresponding rotation matrix.
    double delta_psi = acos(-r_unit.dot(s0_unit)) - acos(r / (2 * s));
    if (abs(delta_psi) < delta_psi_tolerance) return std::nullopt;
    Rotator rotator(s0_unit.cross(r_unit));
    const Vector3d rotated_r = rotator.rotate(r_vec, delta_psi * 180 / M_PI);
    const Vector3d s1 = s0 + rotated_r;

    // Create a Ray object, where the angle now represents |delta_phi|,
    // NOT the goniometer angle.
    return Ray{s1, abs(delta_psi), false};
}

/**
 * @brief Return a Ray object if a prediction is found during a static rotation
 * 
 * @param index The Miller indices of the spot under consideration
 * @param A The (rotation * crystal setting matrix)
 * @param r_setting The setting rotation matrix
 * @param r_setting_inv The inverse of the setting rotation matrix
 * @param s0 The incident beam vector
 * @param m2 The goniometer rotation axis
 * @param rotator The rotator object to generate rotations around axis m2
 * @param dmin The minimum lattice spacing that can be resolved
 * @param phi_beg The angle of the goniometer at the start of the rotation (in degrees)
 * @param d_osc The amount the goniometer rotates during the image capture (in degrees)
 * @return std::array<std::optional<Ray>, 2> 
 */
std::array<std::optional<Ray>, 2> predict_ray_monochromatic_static(
  const std::array<int, 3> &index,
  const Matrix3d &A,
  const Matrix3d &r_setting,
  const Matrix3d &r_setting_inv,
  const Vector3d &s0,
  const Vector3d &m2,
  const Rotator &rotator,
  const double dmin,
  const double phi_beg,
  const double d_osc) {
    const Vector3d hkl_vec{(double)index[0], (double)index[1], (double)index[2]};

    // Calculate the reciprocal space vectors
    const Vector3d r1 = A * hkl_vec;
    const Vector3d s0_rot = r_setting_inv * s0;
    const Vector3d r1_rot = r_setting_inv * r1;
    const double s0_sq = s0.squaredNorm();
    const double r_sq = r1.squaredNorm();
    const double s0pr_sq = (s0 + r1).squaredNorm();

    const double q = m2.dot(s0_rot) * m2.dot(r1_rot);
    const double a = s0_rot.dot(r1_rot) - q;
    const double b = s0_rot.dot(m2.cross(r1_rot));
    const double c = -(r_sq / 2 + q);
    if (a == 0 && b == 0) return {std::nullopt, std::nullopt};

    // Now assume either a or b is non-zero
    const double d = (a != 0) ? atan(b / a) : (b > 0) ? M_PI_2 : -M_PI_2;
    if (c * c > a * a + b * b) return {std::nullopt, std::nullopt};
    const double e = acos(c / sqrt(a * a + b * b));

    // Calculate the angles at which interesction with the Ewald sphere takes place
    // (in degrees). The +180 may be needed to bring the angle into the correct quadrant.
    double angle_first;
    double angle_second;
    if (a >= 0) {
        angle_first = (d - e) * 180 * M_1_PI;
        angle_second = (d + e) * 180 * M_1_PI;
    } else {
        angle_first = (d - e) * 180 * M_1_PI + 180;
        angle_second = (d + e) * 180 * M_1_PI + 180;
    }
    // Bring the angles into the range [0, 360]
    if (angle_first < 0)
        angle_first += 360;
    else if (angle_first > 360)
        angle_first -= 360;
    if (angle_second < 0)
        angle_second += 360;
    else if (angle_second > 360)
        angle_second -= 360;

    if (angle_first > angle_second) std::swap(angle_first, angle_second);

    // Check if the intersection happens within the given rotation. If so, assign a valid
    // Ray object to the corresponding rotation.
    std::optional<Ray> ray_first = std::nullopt;
    std::optional<Ray> ray_second = std::nullopt;
    if (angle_first < d_osc && angle_first > 0) {
        Vector3d r2 = r_setting * rotator.rotate(r1_rot, angle_first);
        Vector3d s1 = s0 + r2;
        bool entering = (s0pr_sq >= s0_sq);
        ray_first = Ray{s1, phi_beg + angle_first, entering};
    }
    if (angle_second < d_osc && angle_second > 0) {
        Vector3d r2 = r_setting * rotator.rotate(r1_rot, angle_second);
        Vector3d s1 = s0 + r2;
        bool entering = (s0pr_sq < s0_sq);
        ray_second = Ray{s1, phi_beg + angle_second, entering};
    }

    return {ray_first, ray_second};
}

/**
 * @brief Return a Ray object if a prediction is found during a scan-varying rotation
 * 
 * @param index The Miller indices of the spot under consideration
 * @param A1 The (rotation * crystal setting matrix) at the start of the rotation
 * @param A2 The (rotation * crystal setting matrix) at the end of the rotation
 * @param s0_1 The incident beam vector at the start of the rotation
 * @param s0_2 The incident beam vector at the end of the rotation
 * @param dmin The minimum lattice spacing that can be resolved
 * @param phi_beg The angle of the goniometer at the start of the rotation (in degrees)
 * @param d_osc The amount the goniometer rotates during the image capture (in degrees)
 * @return std::optional<Ray>
 */
std::optional<Ray> predict_ray_monochromatic_sv(const std::array<int, 3> &index,
                                                const Matrix3d &A1,
                                                const Matrix3d &A2,
                                                const Vector3d &s0_1,
                                                const Vector3d &s0_2,
                                                const double dmin,
                                                const double phi_beg,
                                                const double d_osc) {
    const Vector3d hkl_vec{(double)index[0], (double)index[1], (double)index[2]};

    // Calculate the reciprocal space vectors
    const Vector3d r1 = A1 * hkl_vec;
    const Vector3d r2 = A2 * hkl_vec;
    const Vector3d dr = r2 - r1;
    const Vector3d s0pr1 = s0_1 + r1;
    const Vector3d s0pr2 = s0_2 + r2;

    // Calculate the distances from the Ewald sphere along radii
    const double r1_from_es = s0pr1.norm() - s0_1.norm();
    const double r2_from_es = s0pr2.norm() - s0_2.norm();

    // Check that the reflection cross the ewald sphere and is within
    // the resolution limit
    const bool starts_outside = (r1_from_es >= 0.0);
    const bool ends_outside = (r2_from_es >= 0.0);
    const bool is_outside_res_limit = (r1.squaredNorm() > 1.0 / (dmin * dmin));
    if (starts_outside == ends_outside || is_outside_res_limit) {
        return std::nullopt;
    }

    // Solve the equation |s0_1 + r1 + alpha * dr| = |s0_1| for alpha. This is
    // equivalent to solving the quadratic equation
    //
    // alpha^2*dr.dr + 2*alpha(s0_1 + r1).dr + 2*s0_1.r1 + r1.r1 = 0
    double a = dr.squaredNorm();
    double b = s0pr1.dot(dr);
    double c = r1.squaredNorm() + 2 * s0_1.dot(r1);
    double d = b * b - a * c;
    if (d < 0) return std::nullopt;

    std::pair<double, double> roots1 = {(-b - sqrt(d)) / a, (-b + sqrt(d)) / a};

    // Choose a root that lies in [0,1]
    double alpha1;
    if (0.0 <= roots1.first && roots1.first <= 1.0)
        alpha1 = roots1.first;
    else if (0.0 <= roots1.second && roots1.second <= 1.0)
        alpha1 = roots1.second;
    else
        return std::nullopt;

    // Solve the equation |s0_2 + r2 - alpha * dr| = |s0_2| for alpha. This is
    // equivalent to solving the quadratic equation
    //
    // alpha^2*dr.dr - 2*alpha(s0_2 + r2).dr + 2*s0_2.r2 + r2.r2 = 0
    b = -s0pr2.dot(dr);
    c = r2.squaredNorm() + 2 * s0_2.dot(r2);
    d = b * b - a * c;
    if (d < 0) return std::nullopt;

    std::pair<double, double> roots2 = {(-b - sqrt(d)) / a, (-b + sqrt(d)) / a};

    // Choose a root that lies in [0,1]
    double alpha2;
    if (0.0 <= roots2.first && roots2.first <= 1.0)
        alpha2 = roots2.first;
    else if (0.0 <= roots2.second && roots2.second <= 1.0)
        alpha2 = roots2.second;
    else
        return std::nullopt;

    // Calculate alpha, the fraction along the linear step, as the distance
    // from the Ewald sphere at the start compared to the total distance
    // travelled relative to the Ewald sphere
    double alpha = alpha1 / (alpha1 + alpha2);

    // Linear approximation to the s0 vector at intersection
    Vector3d us0_1 = s0_1.normalized();
    Vector3d us0_at_intersection = alpha * (s0_2.normalized() - us0_1) + us0_1;
    double wavenumber = (s0_1.norm() + s0_2.norm()) * 0.5;
    Vector3d s0_at_intersection = wavenumber * us0_at_intersection;

    // Calculate the scattering vector and rotation angle
    const Vector3d s1 = r1 + alpha * dr + s0_at_intersection;
    const double angle = phi_beg + alpha * d_osc;
    return Ray{s1, angle, starts_outside};
}

// Laue prediction
/**
 * @brief Return a Ray object if a prediction is found for a polychromatic beam.
 * 
 * @param index The Miller indices of the spot under consideration
 * @param A The (rotation * crystal setting matrix)
 * @param s0_unit The unit incident beam vector
 * @param wavelength_min The lower end of the wavelength spectrum
 * @param wavelength_max The upper end of the wavelength spectrum
 * @param dmin The minimum lattice spacing that can be resolved
 * @return std::optional<Ray> 
 */
std::optional<Ray> predict_ray_polychromatic_stills(const std::array<int, 3> &index,
                                                    const Matrix3d &A,
                                                    Vector3d s0_unit,
                                                    const double wavelength_min,
                                                    const double wavelength_max,
                                                    const double dmin) {
    s0_unit.normalize();
    const Vector3d hkl_vec{(double)index[0], (double)index[1], (double)index[2]};

    const Vector3d r = A * hkl_vec;
    double s0 = -r.norm() / (2 * r.normalized().dot(s0_unit));
    if ((1 / wavelength_max > s0) || (s0 > 1 / wavelength_min) || (1 / dmin < s0))
        return std::nullopt;

    const Vector3d s1 = s0 * s0_unit + r;
    return Ray{s1, 0.0, false};
}

/**
 * @brief Return a Ray object if a prediction is found during a scan-varying rotation for a polychromatic beam.
 * 
 * @param index The Miller indices of the spot under consideration
 * @param A1 The (rotation * crystal setting matrix) at the start of the rotation
 * @param A2 The (rotation * crystal setting matrix) at the end of the rotation
 * @param s0_1_unit The unit incident beam vector at the start of the rotation
 * @param s0_2_unit The unit incident beam vector at the end of the rotation
 * @param wavelength_min The lower end of the wavelength spectrum
 * @param wavelength_max The upper end of the wavelength spectrum
 * @param dmin The minimum lattice spacing that can be resolved
 * @param phi_beg The angle of the goniometer at the start of the rotation (in degrees)
 * @param d_osc The amount the goniometer rotates during the image capture (in degrees)
 * @return std::optional<Ray> 
 */
std::optional<Ray> predict_ray_polychromatic_rotational(const std::array<int, 3> &index,
                                                        const Matrix3d &A1,
                                                        const Matrix3d &A2,
                                                        Vector3d s0_1_unit,
                                                        Vector3d s0_2_unit,
                                                        const double wavelength_min,
                                                        const double wavelength_max,
                                                        const double dmin,
                                                        const double phi_beg,
                                                        const double d_osc) {
    // FIXME: Not implemented
    return std::nullopt;
}
