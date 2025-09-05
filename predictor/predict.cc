/**
 * @file predict.cc
 * @brief This program implements the algorithm for spot prediction.
 *
 * ReekeIndexGenerator outlined in in the LURE workshop notes.
 *
 */

#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <chrono>
#include <cmath>
#include <common.hpp>
#include <concepts>
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <dx2/scan.hpp>
#include <exception>
#include <fstream>
#include <gemmi/symmetry.hpp>
#include <iostream>  // Debugging
#include <nlohmann/json.hpp>
#include <thread>
#include <type_traits>
#include <vector>

using json = nlohmann::json;

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;

// Enums to specify information about the experiment and beam.
enum class ExperimentType { Stills, Rotational };
enum class RotationalType { Static, ScanVarying };
enum class BeamType { Monochromatic, Polychromatic };

struct ScanVaryingType {
    bool beam = false;
    bool crystal = false;
    bool r_setting = false;
};

struct PredictionType {
    ExperimentType experiment_type = ExperimentType::Rotational;
    RotationalType rotational_type = RotationalType::ScanVarying;
};

#pragma region Utilities
/**
 * @brief A class to store the axis of rotation and return a rotation matrix for a given angle using the Rodriguez formula.
 * 
 */
class Rotator {
  private:
    Vector3d axis_ = Vector3d{0, 0, 0};

  public:
    Rotator(const Vector3d& axis) : axis_(axis.normalized()) {}

    /**
	 * @brief Output a rotation matrix corresponding to a given axis and angle.
	 *
	 * @param θ The angle of rotation (in degrees)
	 */
    Matrix3d rotation_matrix(double θ) const {
        // Convert to radians
        θ = θ * M_PI / 180;
        return Matrix3d{{cos(θ) + axis_(0) * axis_(0) * (1 - cos(θ)),
                         -axis_(2) * sin(θ) + axis_(0) * axis_(1) * (1 - cos(θ)),
                         axis_(1) * sin(θ) + axis_(0) * axis_(1) * (1 - cos(θ))},
                        {axis_(2) * sin(θ) + axis_(0) * axis_(1) * (1 - cos(θ)),
                         cos(θ) + axis_(1) * axis_(1) * (1 - cos(θ)),
                         -axis_(0) * sin(θ) + axis_(1) * axis_(2) * (1 - cos(θ))},
                        {-axis_(1) * sin(θ) + axis_(0) * axis_(2) * (1 - cos(θ)),
                         axis_(0) * sin(θ) + axis_(1) * axis_(2) * (1 - cos(θ)),
                         cos(θ) + axis_(2) * axis_(2) * (1 - cos(θ))}};
    }

    /**
	 * @brief Rotate a 3D vector by a given angle around the pre-specified axis
	 *
	 */
    Vector3d rotate(const Vector3d& vec, double θ) const {
        return rotation_matrix(θ) * vec;
    }

    /**
	 * @brief Multiply the rotation matrix (angle θ) by a 3x3 matrix and return the result.
	 *
	 */
    Matrix3d rotate(const Matrix3d& mat, double θ) const {
        return rotation_matrix(θ) * mat;
    }
};

/**
 * @brief Takes in a json list of numbers and returns a 3x3 matrix representation of it.
 * 
 * @param matrix_json The json object, a list of numbers with length 9.
 * @return Matrix3d 
 */
Matrix3d matrix_3d_from_json(json matrix_json) {
    return Matrix3d{{matrix_json[0], matrix_json[1], matrix_json[2]},
                    {matrix_json[3], matrix_json[4], matrix_json[5]},
                    {matrix_json[6], matrix_json[7], matrix_json[8]}};
}

/**
 * @brief Takes in a json list of numbers and returns a 3D vector representation of it.
 * 
 * @param vector_json The json object, a list of numbers with length 3.
 * @return Vector3d 
 */
Vector3d vector_3d_from_json(json vector_json) {
    return Vector3d{{vector_json[0], vector_json[1], vector_json[2]}};
}

/**
 * @brief A struct to store information about the predicted reflected ray.
 * 
 */
struct Ray {
    Vector3d s1;
    double angle;
    bool entering;
};
#pragma endregion

#pragma region Argument Parser Configuration
/**
 * @brief Take a default-initialized ArgumentParser object and configure it
 *      with the arguments to be parsed; assign various properties to each
 *      argument, eg. help message, default value, etc.
 *
 * @param parser The ArgumentParser object (pre-input) to be configured.
 */
void configure_parser(argparse::ArgumentParser& parser) {
    parser.add_argument("-e", "--expt")
      .help("path to DIALS expt file")
      .nargs(argparse::nargs_pattern::at_least_one);  //.required();
    parser.add_argument("--dmin")
      .help("minimum d-spacing of predicted reflections")
      .scan<'f', double>()
      .default_value(-1.0);
    //.required();
    parser.add_argument("-s", "--force_static")
      .help("for a scan varying model, forces static prediction")
      .default_value(false)
      .implicit_value(true);
    // The below is the opposite of ignore_shadows used in DIALS
    // This configuration allows for natural implicit-value flagging.
    parser.add_argument("-d", "--dynamic_shadows")
      .help("enables dynamic shadowing")
      .default_value(false)
      .implicit_value(true);
    parser.add_argument("-b", "--buffer_size")
      .help(
        "calculates predictions within a buffer zone of n images either side"
        "of the scan")
      .scan<'i', int>()
      .default_value<int>(0);
    parser.add_argument("-n", "--nthreads")
      .help(
        "the number of threads to use for the fft calculation, "
        "defaults to the value of std::thread::hardware_concurrency, "
        "better performance can typically be obtained with a higher number"
        "of threads than this. UNUSED.")
      .scan<'u', size_t>()
      .default_value<size_t>(std::thread::hardware_concurrency());
}

/**
 * @brief Take an ArgumentParser object after the user has entered input and check
 *      it for consistency; output errors and exit the program if a check fails.
 *
 * @param parser The ArgumentParser object (post-input) to be verified.
 */
void verify_arguments(const argparse::ArgumentParser& parser) {
    if (!parser.is_used("expt")) {
        logger.error("Must specify experiment list file with -e or --expt\n");
        std::exit(1);
    }
    if (parser.is_used("buffer_size") && parser.get<int>("buffer_size") < 0) {
        logger.error("--buffer_size cannot be negative\n");
    }
    if (parser.is_used("nthreads") && parser.get<size_t>("nthreads") < 1) {
        logger.error("--nthreads cannot be less than 1\n");
        std::exit(1);
    }
}
#pragma endregion

#pragma region Index Generators
/**
 * A class to generate miller indices for rotational experiments using the Reeke algorithm.
 */
class ReekeIndexGenerator {
  public:
    ReekeIndexGenerator(const Matrix3d& A1,
                        const Matrix3d& A2,
                        gemmi::GroupOps& crystal_symmetry_operations,
                        const Vector3d& s0_1,
                        const Vector3d& s0_2,
                        const double dmin,
                        const bool use_monochromatic)
        : A1(A1),
          A2(A2),
          s0_1(s0_1),
          s0_2(s0_2),
          dmin(dmin),
          use_monochromatic(use_monochromatic),
          crystal_symmetry_operations(crystal_symmetry_operations) {
        auto P1 = MatrixXd{
          {A1(0, 0), A1(0, 1), A1(0, 2), s0_1[0]},
          {A1(1, 0), A1(1, 1), A1(1, 2), s0_1[1]},
          {A1(2, 0), A1(2, 1), A1(2, 2), s0_1[2]},
        };
        auto P2 = MatrixXd{
          {A2(0, 0), A2(0, 1), A2(0, 2), s0_2[0]},
          {A2(1, 0), A2(1, 1), A2(1, 2), s0_2[1]},
          {A2(2, 0), A2(2, 1), A2(2, 2), s0_2[2]},
        };
        T1 = P1.transpose() * P1;
        T2 = P2.transpose() * P2;
    }

    // Generate and return the next miller index.
    std::optional<std::array<int, 3>> next() {
        const int enter = 0;
        const int yield = 1;

        // Static variables
        static std::optional<std::pair<int, int>> h_lims;
        static std::optional<std::pair<int, int>> k_lims;
        static std::array<std::optional<std::pair<int, int>>, 2> l_lims_arr;
        static std::size_t l_index;
        static int state = enter;

        // The first time the function is executed, control starts at the top (case 0).
        // On subsequent calls, control starts after the "yield" point (case 1), The
        // static variables ensure that the state is recovered on each subsequent
        // function call.
        std::array<int, 3> result;
        switch (state) {
        case enter:
            state = yield;
            h_lims = calc_h_limits();
            if (!h_lims) break;
            for (; h_lims->first <= h_lims->second; h_lims->first++) {
                k_lims = calc_k_limits(h_lims->first);
                if (!k_lims) continue;
                for (; k_lims->first <= k_lims->second; k_lims->first++) {
                    l_lims_arr = calc_l_limits(h_lims->first, k_lims->first);
                    l_index = 0;
                    for (; l_index < 2; ++l_index) {
                        if (!l_lims_arr[l_index]) continue;
                        for (;
                             l_lims_arr[l_index]->first <= l_lims_arr[l_index]->second;
                             l_lims_arr[l_index]->first++) {
                            result = std::array<int, 3>{
                              h_lims->first, k_lims->first, l_lims_arr[l_index]->first};
                            if (!crystal_symmetry_operations.is_systematically_absent(
                                  result)) {
                                return result;
                            case yield:;
                            }
                        }
                    }
                }
            }
        }
        state = enter;
        return std::nullopt;
    }

  private:
    /**
     * @brief Get the min and max elements from among the passed in values (if any).
     * 
     * @param pair1 
     * @param pair2 
     * @return std::optional<std::pair<T, T>> 
     */
    template <typename T>
        requires(std::integral<T> or std::floating_point<T>)
    auto get_min_max_pair(std::optional<std::pair<T, T>> pair1,
                          std::optional<std::pair<T, T>> pair2)
      -> std::optional<std::pair<T, T>> {
        T min_v;
        T max_v;
        if (!pair1 && !pair2) return std::nullopt;
        if (pair1) {
            min_v = std::min(pair1->first, pair1->second);
            max_v = std::max(pair1->first, pair1->second);
            if (pair2) {
                min_v = std::min(min_v, std::min(pair2->first, pair2->second));
                max_v = std::max(max_v, std::max(pair2->first, pair2->second));
            }
        } else {
            min_v = std::min(pair2->first, pair2->second);
            max_v = std::max(pair2->first, pair2->second);
        }
        return std::pair<T, T>{min_v, max_v};
    }

    /**
	 * @brief Calculate the looping limits of the Miller index h, considering only the resolution sphere
	 *
	 * @param a The vector normal to the plane of constant h, pointing in the direction of increasing h
	 * @param s0 The incident beam vector
	 * @param dmin The minimum lattice spacing that can be resolved
	 * @return std::pair<double, double>
	 */
    auto calc_h_limits_resolution(const Vector3d& a, const Vector3d& s0)
      -> std::pair<double, double> {
        double dstar_max = 1.0 / dmin;
        double s0_len_sq = s0.squaredNorm();
        double s0_dot_a = s0.dot(a);

        double e = -dstar_max * dstar_max * s0_dot_a / (2 * s0_len_sq);
        double f =
          dstar_max * sqrt(std::max(0.0, 1 - dstar_max * dstar_max / (4 * s0_len_sq)));

        return {e - f, e + f};
    }

    /**
	 * @brief Calculate the looping limits of the Miller index h
	 *
	 * @param A1 The (rotation * crystal setting matrix) at the start of the rotation
	 * @param A2 The (rotation * crystal setting matrix) at the end of the rotation
	 * @param s0_1 The incident beam vector at the start of the rotation
	 * @param s0_2 The incident beam vector at the end of the rotation
	 * @param dmin The minimum lattice spacing that can be resolved
	 * @return std::optional<std::pair<int, int>>
	 */
    auto calc_h_limits() -> std::optional<std::pair<int, int>> {
        const Vector3d a1 = A1.inverse().row(0);
        const Vector3d a2 = A2.inverse().row(0);
        const double a1_len = a1.norm();
        const double a2_len = a2.norm();
        const double s0_1_len = s0_1.norm();
        const double s0_2_len = s0_2.norm();
        const double s0_1_dot_a1 = s0_1.dot(a1);
        const double s0_2_dot_a2 = s0_2.dot(a2);

        // Calculate Ewald limits
        std::optional<std::pair<double, double>> h_limits_1 = std::pair<double, double>{
          -a1_len * s0_1_len - s0_1_dot_a1, a1_len * s0_1_len - s0_1_dot_a1};
        std::optional<std::pair<double, double>> h_limits_2 = std::pair<double, double>{
          -a2_len * s0_2_len - s0_2_dot_a2, a2_len * s0_2_len - s0_2_dot_a2};

        // Calculate resolution limits
        std::pair<double, double> h_limits_resolution_1 =
          calc_h_limits_resolution(a1, s0_1);
        std::pair<double, double> h_limits_resolution_2 =
          calc_h_limits_resolution(a2, s0_2);

        // Conditionally combine the Ewald and resolution limits
        // The logic here is that is if the point of tangency between a plane of constant h and the Ewald sphere lies
        // outside the resolution sphere, we use the corresponding resolution limit. Otherwise, we keep the Ewald limit.
        if (2 * (s0_1_len * s0_1_len + abs(s0_1_len * s0_1_dot_a1) / a1_len)
            > 1 / (dmin * dmin))
            h_limits_1->first = h_limits_resolution_1.first;
        if (2 * (s0_1_len * s0_1_len - abs(s0_1_len * s0_1_dot_a1) / a1_len)
            > 1 / (dmin * dmin))
            h_limits_1->second = h_limits_resolution_1.second;
        if (2 * (s0_2_len * s0_2_len + abs(s0_2_len * s0_2_dot_a2) / a2_len)
            > 1 / (dmin * dmin))
            h_limits_2->first = h_limits_resolution_2.first;
        if (2 * (s0_2_len * s0_2_len - abs(s0_2_len * s0_2_dot_a2) / a2_len)
            > 1 / (dmin * dmin))
            h_limits_2->second = h_limits_resolution_2.second;

        // Verify that the combined start and end limits are in the correct order
        if (h_limits_1->first > h_limits_1->second) h_limits_1 = std::nullopt;
        if (h_limits_2->first > h_limits_2->second) h_limits_2 = std::nullopt;

        auto h_min_max = get_min_max_pair(h_limits_1, h_limits_2);
        if (!h_min_max) return std::nullopt;

        std::pair<double, double> h_limits = h_min_max.value();

        return std::pair{(int)h_limits.first, (int)h_limits.second + 1};
    }

    /**
	 * @brief Calculate the looping limits of the Miller index k, given index h and considering only the Ewald sphere
	 *
	 * @param T A 4d matrix; modified from the one defined on p. 54 of LURE Phase 1 and 2.
	 * @return std::optional<std::pair<int, int>>
	 */
    auto calc_k_limits_ewald(const Matrix4d& T, const int h)
      -> std::optional<std::pair<int, int>> {
        double r0 = T(2, 3) * T(2, 3)
                    + h
                        * (2 * (T(0, 2) * T(2, 3) - T(0, 3) * T(2, 2))
                           + h * (T(0, 2) * T(0, 2) - T(0, 0) * T(2, 2)));
        double r1 = T(1, 2) * T(2, 3) - T(1, 3) * T(2, 2)
                    + h * (T(0, 2) * T(1, 2) - T(0, 1) * T(2, 2));
        double r2 = T(1, 2) * T(1, 2) - T(1, 1) * T(2, 2);

        if (r2 == 0) return std::nullopt;

        double d = r1 * r1 - r0 * r2;
        if (d < 0) return std::nullopt;

        int a = int((-r1 + sqrt(d)) / r2);
        int b = int((-r1 - sqrt(d)) / r2) + 1;
        return std::pair<int, int>{a, b};
    }

    /**
     * @brief Calculate the looping limits of the Miller index k, given index h and considering only the resolution sphere
     * 
     * @param T A 4d matrix; modified from the one defined on p. 54 of LURE Phase 1 and 2.
     * @param h The h Miller index
     * @param dmin The minimum lattice spacing that can be resolved
     * @return std::optional<std::pair<int, int>> 
     */
    auto calc_k_limits_resolution(const Matrix4d& T, const int h)
      -> std::optional<std::pair<int, int>> {
        double r0 =
          h * h * (T(0, 2) * T(0, 2) - T(0, 0) * T(2, 2)) + T(2, 2) / (dmin * dmin);
        double r1 = h * (T(0, 2) * T(1, 2) - T(0, 1) * T(2, 2));
        double r2 = T(1, 2) * T(1, 2) - T(1, 1) * T(2, 2);

        if (r2 == 0) return std::nullopt;

        double d = r1 * r1 - r0 * r2;
        if (d < 0) return std::nullopt;

        int a = int((-r1 + sqrt(d)) / r2);
        int b = int((-r1 - sqrt(d)) / r2) + 1;
        return std::pair<int, int>{a, b};
    }

    /**
     * @brief Calculate the looping limits of the Miller index k, given index h.
     * 
     * @param T1 A 4d matrix (start of rotation); modified from the one defined on p. 54 of LURE Phase 1 and 2.
     * @param T2 A 4d matrix (end of rotation); modified from the one defined on p. 54 of LURE Phase 1 and 2.
     * @param h The h Miller index
     * @param dmin The minimum lattice spacing that can be resolved
     * @return std::optional<std::pair<int, int>> 
     */
    auto calc_k_limits(const int h) -> std::optional<std::pair<int, int>> {
        std::optional<std::pair<int, int>> k_limits_ewald_1 =
          calc_k_limits_ewald(T1, h);
        std::optional<std::pair<int, int>> k_limits_ewald_2 =
          calc_k_limits_ewald(T2, h);
        std::optional<std::pair<int, int>> k_limits_resolution =
          calc_k_limits_resolution(T1, h);

        if (!k_limits_resolution) return std::nullopt;

        // Find the min and max limit values, if they exist.
        // Otherwise, return std::nullopt as no diffraction occurs
        auto k_min_max = get_min_max_pair(k_limits_ewald_1, k_limits_ewald_2);
        if (!k_min_max) return std::nullopt;

        std::pair<int, int> k_limits = k_min_max.value();

        k_limits.first = std::max(k_limits.first, k_limits_resolution->first);
        k_limits.second = std::min(k_limits.second, k_limits_resolution->second);

        return k_limits;
    }

    /**
     * @brief Calculate the looping limits of the Miller index l, given indices h and k and considering only the Ewald sphere
     * 
     * @param T A 4d matrix; modified from the one defined on p. 54 of LURE Phase 1 and 2.
     * @param h The h Miller index
     * @param k The k Miller index
     * @return std::optional<std::pair<int, int>> 
     */
    auto calc_l_limits_ewald(const Matrix4d& T, const int h, const int k)
      -> std::optional<std::pair<int, int>> {
        double q0 = T(0, 0) * h * h + 2 * T(0, 1) * h * k + T(1, 1) * k * k
                    + 2 * T(0, 3) * h + 2 * T(1, 3) * k;
        double q1 = T(0, 2) * h + T(1, 2) * k + T(2, 3);
        double q2 = T(2, 2);

        if (q2 == 0) return std::nullopt;

        double d = q1 * q1 - q0 * q2;
        if (d < 0) return std::nullopt;

        int a = int((-q1 - sqrt(d)) / q2);
        int b = int((-q1 + sqrt(d)) / q2) + 1;
        return std::pair<int, int>{a, b};
    }

    /**
     * @brief Calculate the looping limits of the Miller index l, given indices h and k and considering only the resolution sphere
     * 
     * @param T A 4d matrix; modified from the one defined on p. 54 of LURE Phase 1 and 2.
     * @param h The h Miller index
     * @param k The k Miller index
     * @param dmin The minimum lattice spacing that can be resolved
     * @return std::optional<std::pair<int, int>> 
     */
    auto calc_l_limits_resolution(const Matrix4d& T, const int h, const int k)
      -> std::optional<std::pair<int, int>> {
        double q0 =
          T(0, 0) * h * h + 2 * T(0, 1) * h * k + T(1, 1) * k * k - 1.0 / (dmin * dmin);
        double q1 = T(0, 2) * h + T(1, 2) * k;
        double q2 = T(2, 2);

        if (q2 == 0) return std::nullopt;

        double d = q1 * q1 - q0 * q2;
        if (d < 0) return std::nullopt;

        int a = int((-q1 - sqrt(d)) / q2);
        int b = int((-q1 + sqrt(d)) / q2) + 1;
        return std::pair<int, int>{a, b};
    }

    /**
 * @brief Calculate the looping limits of the Miller index l, given indices h and k.
 *
 * @param T1 The T-matrix at the start of the rotation
 * @param T2 The T-matrix at the end of the rotation
 * @param h The h Miller index
 * @param k The k Miller index
 * @param dmin The minimum lattice spacing that can be resolved
 * @return std::array<std::optional<std::pair<int, int>>, 2>
 */
    auto calc_l_limits(const int h, const int k)
      -> std::array<std::optional<std::pair<int, int>>, 2> {
        std::optional<std::pair<int, int>> l_limits_ewald_1 =
          calc_l_limits_ewald(T1, h, k);
        std::optional<std::pair<int, int>> l_limits_ewald_2 =
          calc_l_limits_ewald(T2, h, k);
        std::optional<std::pair<int, int>> l_limits_resolution =
          calc_l_limits_resolution(T1, h, k);
        if (!l_limits_resolution) return {};

        // Rearrange the results into a vector of size 1 or 2, depending on the results of the ewald calculations
        std::array<std::optional<std::pair<int, int>>, 2> l_limits_ewald;
        if (use_monochromatic) {
            // This is the vast majority of experiments and an optimisation exists when both
            // l_limits_ewald_1 and l_limits_ewald_2 are valid, in which we only need to consider
            // the thin slices of l around the min of the two and around the max of the two.
            if (l_limits_ewald_1 && l_limits_ewald_2) {
                l_limits_ewald[0] = std::pair{
                  std::min(l_limits_ewald_1->first, l_limits_ewald_2->first),
                  std::max(l_limits_ewald_1->first, l_limits_ewald_2->first) + 1};
                l_limits_ewald[1] = {
                  std::min(l_limits_ewald_1->second, l_limits_ewald_2->second) - 1,
                  std::max(l_limits_ewald_1->second, l_limits_ewald_2->second)};
            } else if (l_limits_ewald_1)
                l_limits_ewald[0] = l_limits_ewald_1.value();
            else if (l_limits_ewald_2)
                l_limits_ewald[1] = l_limits_ewald_2.value();
            else
                return {};
        } else {
            if (l_limits_ewald_1)
                l_limits_ewald[0] = l_limits_ewald_1.value();
            else if (l_limits_ewald_2)
                l_limits_ewald[1] = (l_limits_ewald_2.value());
            else
                return {};
        }

        // Rearrange the results into an array of size 2 with 0, 1, or 2 valid values.
        std::array<std::optional<std::pair<int, int>>, 2> l_limits;
        for (std::size_t i = 0; i < l_limits_ewald.size(); ++i) {
            if (!l_limits_ewald[i]) continue;
            if (l_limits_resolution->first > l_limits_ewald[i]->first) {
                l_limits_ewald[i]->first = l_limits_resolution->first;
            }
            if (l_limits_resolution->second < l_limits_ewald[i]->second) {
                l_limits_ewald[i]->second = l_limits_resolution->second;
            }
            if (l_limits_ewald[i]->first < l_limits_ewald[i]->second) {
                l_limits[i] = std::pair<int, int>{l_limits_ewald[i]->first,
                                                  l_limits_ewald[i]->second};
            }
        }

        // Ensure that if there are two ranges, they are ordered and non-overlapping
        if (l_limits[0] && l_limits[1]) {
            if (l_limits[0]->first > l_limits[1]->first) {
                std::swap(l_limits[0], l_limits[1]);
            }
            if (l_limits[1]->first <= l_limits[0]->second) {
                l_limits[0]->second =
                  std::max(l_limits[0]->second, l_limits[1]->second);
                l_limits[1] = std::nullopt;
            }
        }

        return l_limits;
    }

    Matrix3d A1;
    Matrix3d A2;
    Matrix4d T1;
    Matrix4d T2;
    Vector3d s0_1;
    Vector3d s0_2;
    double dmin;
    bool use_monochromatic;
    gemmi::GroupOps crystal_symmetry_operations;
};

class PolychromaticRotationalIndexGenerator {
    // FIXME: Currently, the Reek index generator has a use_monochromatic boolean that, when set to false,
    // can generate indices for polychromtic beams. This is, however, expensive because the fine-slicing
    // around the Ewald sphere no longer works and the full range of indices inside the larger Ewald sphere
    // have to be generated. A new index generator is here needed, it should generate the indixes between
    // the inner and outer Ewald spheres.
    // Alternatively, the StillsIndex generator may be used (angular tolerance = rotation angle) without
    // (IMO, but this is unmeasured) too much of an additional cost, since the number of polychromatic spots
    // is large anyway, and the added
};

/**
 * A class to generate Miller indices for stills experiments
 */
class StillsIndexGenerator {
  public:
    // FIXME: This is quite ugly to accommodate polychromatic prediction, maybe a separate generator
    // or better naming of variables will solve the problem.
    StillsIndexGenerator(const Matrix3d& A,
                         gemmi::GroupOps& crystal_symmetry_operations,
                         const Vector3d& s0_upper,
                         const Vector3d& s0_lower,
                         const double angular_tolerance)
        : A(A),
          s0(s0_upper),
          s0_lower(s0_lower),
          crystal_symmetry_operations(crystal_symmetry_operations) {
        s0_len_sq = s0.squaredNorm();
        s0_len_sq_min =
          s0_lower.squaredNorm() * (1 - angular_tolerance) * (1 - angular_tolerance);
        s0_len_sq_max = s0_len_sq * (1 + angular_tolerance) * (1 + angular_tolerance);
        auto P = MatrixXd{
          {A(0, 0), A(0, 1), A(0, 2), s0[0]},
          {A(1, 0), A(1, 1), A(1, 2), s0[1]},
          {A(2, 0), A(2, 1), A(2, 2), s0[2]},
        };
        auto P_inner = MatrixXd{
          {A(0, 0), A(0, 1), A(0, 2), s0_lower[0]},
          {A(1, 0), A(1, 1), A(1, 2), s0_lower[1]},
          {A(2, 0), A(2, 1), A(2, 2), s0_lower[2]},
        };
        T = P.transpose() * P;
        T_inner = P_inner.transpose() * P_inner;
    }

    // Generate and return the next miller index.
    std::optional<std::array<int, 3>> next() {
        // Constants to make clearer
        const int enter = 0;
        const int yield = 1;

        // Static variables
        static std::optional<std::pair<int, int>> h_lims;
        static std::optional<std::pair<int, int>> k_lims;
        static std::array<std::optional<std::pair<int, int>>, 2> l_lims_arr;
        static std::size_t l_index;
        static int state = enter;

        // This switch simulates a co-routine or python generator. The first time
        // the function is executed, control starts at the top (case 0). On
        // subsequent calls, control starts after the "yield" point (case 1), The
        // static variables ensure that the state is recovered on each subsequent
        // function call.
        std::array<int, 3> result;
        switch (state) {
        case enter:
            state = yield;
            h_lims = calc_h_limits();
            if (!h_lims) break;
            for (; h_lims->first <= h_lims->second; h_lims->first++) {
                k_lims = calc_k_limits(h_lims->first);
                if (!k_lims) continue;
                for (; k_lims->first <= k_lims->second; k_lims->first++) {
                    l_lims_arr = calc_l_limits(h_lims->first, k_lims->first);
                    l_index = 0;
                    for (; l_index < 2; ++l_index) {
                        if (!l_lims_arr[l_index]) continue;
                        for (;
                             l_lims_arr[l_index]->first <= l_lims_arr[l_index]->second;
                             l_lims_arr[l_index]->first++) {
                            result = std::array<int, 3>{
                              h_lims->first, k_lims->first, l_lims_arr[l_index]->first};
                            if (!crystal_symmetry_operations.is_systematically_absent(
                                  result)) {
                                return result;
                            case yield:;
                            }
                        }
                    }
                }
            }
        }
        state = enter;
        return std::nullopt;
    }

  private:
    /**
	 * @brief Calculate the looping limits of the Miller index h
	 *
	 * @return std::optional<std::pair<int, int>>
	 */
    auto calc_h_limits() -> std::optional<std::pair<int, int>> {
        const Vector3d a = A.inverse().row(0);
        const double a_len = a.norm();
        const double s0_len_max = sqrt(s0_len_sq_max);
        const double s0_dot_a = s0.dot(a);

        // Calculate Ewald limits
        std::optional<std::pair<double, double>> h_limits = std::pair<double, double>{
          -s0_dot_a - a_len * s0_len_max, -s0_dot_a + a_len * s0_len_max};

        if (!h_limits) return std::nullopt;
        return std::pair{(int)h_limits->first, (int)h_limits->second + 1};
    }

    /**
     * @brief Calculate the looping limits of the Miller index k
     * 
     * @param h The h Miller index
     * @return std::optional<std::pair<int, int>> 
     */
    auto calc_k_limits(const int h) -> std::optional<std::pair<int, int>> {
        double r0 = T(2, 3) * T(2, 3)
                    + h
                        * (2 * (T(0, 2) * T(2, 3) - T(0, 3) * T(2, 2))
                           + h * (T(0, 2) * T(0, 2) - T(0, 0) * T(2, 2)))
                    + T(2, 2) * (s0_len_sq_max - s0_len_sq);
        double r1 = T(1, 2) * T(2, 3) - T(1, 3) * T(2, 2)
                    + h * (T(0, 2) * T(1, 2) - T(0, 1) * T(2, 2));
        double r2 = T(1, 2) * T(1, 2) - T(1, 1) * T(2, 2);

        if (r2 == 0) return std::nullopt;

        double d = r1 * r1 - r0 * r2;
        if (d < 0) return std::nullopt;

        int a = int((-r1 + sqrt(d)) / r2);
        int b = int((-r1 - sqrt(d)) / r2) + 1;
        return std::pair<int, int>{a, b};
    }

    /**
     * @brief Calculate the looping limits of the Miller index l, given indices h and k and considering only the Ewald sphere
     * 
     * @param T A 4d matrix; modified from the one defined on p. 54 of LURE Phase 1 and 2.
     * @param h The h Miller index
     * @param k The k Miller index
     * @return std::array<std::optional<std::pair<double, double>>, 2>
     */
    auto calc_l_limits_ewald(const int h, const int k)
      -> std::array<std::optional<std::pair<double, double>>, 2> {
        std::array<std::optional<std::pair<double, double>>, 2> l_limits;

        double q0 = T(0, 0) * h * h + 2 * T(0, 1) * h * k + T(1, 1) * k * k
                    + 2 * T(0, 3) * h + 2 * T(1, 3) * k - s0_len_sq_max + s0_len_sq;
        double q1 = T(0, 2) * h + T(1, 2) * k + T(2, 3);
        double q2 = T(2, 2);

        if (q2 == 0) return {std::nullopt, std::nullopt};

        double d = q1 * q1 - q0 * q2;
        if (d < 0)
            l_limits[0] = std::nullopt;
        else {
            double a = int((-q1 - sqrt(d)) / q2);
            double b = int((-q1 + sqrt(d)) / q2);
            l_limits[0] = std::pair<double, double>{a, b};
        }

        q0 = T_inner(0, 0) * h * h + 2 * T_inner(0, 1) * h * k + T_inner(1, 1) * k * k
             + 2 * T_inner(0, 3) * h + 2 * T_inner(1, 3) * k - s0_len_sq_min;
        q1 = T_inner(0, 2) * h + T_inner(1, 2) * k + T_inner(2, 3);
        q2 = T_inner(2, 2);

        d = q1 * q1 - q0 * q2;
        if (d < 0)
            l_limits[1] = std::nullopt;
        else {
            double a = int((-q1 - sqrt(d)) / q2);
            double b = int((-q1 + sqrt(d)) / q2);
            l_limits[1] = std::pair<double, double>{a, b};
        }

        return l_limits;
    }

    /**
     * @brief Calculate the looping limits of the Miller index l, given indices h and k.
     *
     * @param h The h Miller index
     * @param k The k Miller index
     * @return std::array<std::optional<std::pair<int, int>>, 2>
     */
    auto calc_l_limits(const int h, const int k)
      -> std::array<std::optional<std::pair<int, int>>, 2> {
        std::array<std::optional<std::pair<double, double>>, 2> l_limits_ewald =
          calc_l_limits_ewald(h, k);

        std::optional<std::pair<double, double>> l_limits_outer = l_limits_ewald[1];
        if (!l_limits_outer) return {std::nullopt, std::nullopt};

        std::optional<std::pair<double, double>> l_limits_inner = l_limits_ewald[0];

        std::optional<std::pair<double, double>> l_limits_1 = l_limits_outer.value();
        std::optional<std::pair<double, double>> l_limits_2 = l_limits_outer.value();
        std::array<std::optional<std::pair<int, int>>, 2> l_limits;

        if (l_limits_inner) {
            if (l_limits_inner->first < l_limits_1->second)
                l_limits_1->second = l_limits_inner->first;
            if (l_limits_inner->second > l_limits_2->first)
                l_limits_2->first = l_limits_inner->second;

            if (l_limits_1->first > l_limits_1->second) l_limits_1 = std::nullopt;
            if (l_limits_2->first > l_limits_2->second) l_limits_2 = std::nullopt;
        }

        if (l_limits_1)
            l_limits[0] =
              std::pair{int(l_limits_1->first), int(l_limits_1->second) + 1};
        if (l_limits_2)
            l_limits[1] =
              std::pair{int(l_limits_2->first), int(l_limits_2->second) + 1};

        if (l_limits[0] && l_limits[1])
            if (l_limits[0]->second > l_limits[1]->first) {
                l_limits[0]->second = l_limits[1]->second;
                l_limits[1] = std::nullopt;
            }
        return l_limits;
    }

    Matrix3d A;
    Matrix4d T;
    Matrix4d T_inner;
    Vector3d s0;
    Vector3d s0_lower;
    double s0_len_sq;
    double s0_len_sq_min;
    double s0_len_sq_max;
    gemmi::GroupOps crystal_symmetry_operations;
};
#pragma endregion

#pragma region Ray Predictors
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
std::optional<Ray> predict_ray_monochromatic_stills(const std::array<int, 3>& index,
                                                    const Matrix3d& A,
                                                    const Vector3d& s0,
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
  const std::array<int, 3>& index,
  const Matrix3d& A,
  const Matrix3d& r_setting,
  const Matrix3d& r_setting_inv,
  const Vector3d& s0,
  const Vector3d& m2,
  const Rotator& rotator,
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
std::optional<Ray> predict_ray_monochromatic_sv(const std::array<int, 3>& index,
                                                const Matrix3d& A1,
                                                const Matrix3d& A2,
                                                const Vector3d& s0_1,
                                                const Vector3d& s0_2,
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
std::optional<Ray> predict_ray_polychromatic_stills(const std::array<int, 3>& index,
                                                    const Matrix3d& A,
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
std::optional<Ray> predict_ray_polychromatic_rotational(const std::array<int, 3>& index,
                                                        const Matrix3d& A1,
                                                        const Matrix3d& A2,
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

#pragma endregion

int main(int argc, char** argv) {
    auto t1 = std::chrono::system_clock::now();
    auto parser = argparse::ArgumentParser();
    configure_parser(parser);

    // Parse the command-line input against the defined parser
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        logger.error(err.what());
        std::exit(1);
    }

    verify_arguments(parser);

    // Obtain argument values from the parsed command-line input
    const auto param_expt_paths = parser.get<std::vector<std::string>>("expt");
    auto param_dmin = parser.get<double>("dmin");
    auto param_force_static = parser.get<bool>("force_static");
    const auto param_dynamic_shadows = parser.get<bool>("dynamic_shadows");
    const auto param_buffer_size = parser.get<int>("buffer_size");
    const auto param_nthreads = parser.get<size_t>("nthreads");
    const std::string output_file_path = "predicted.refl";
    const uint64_t predicted_flag = (1 << 0);

#pragma region Create Reflection Data Containers
    // Create std::vectors to store results in, and later add them as columns to a ReflectionTable.
    ReflectionTable predicted;
    // Shape {size, 3}
    std::vector<int32_t> hkl;
    std::vector<double> s1;
    std::vector<double> xyz_px;
    std::vector<double> xyz_mm;
    std::vector<double> s0_cal;
    // Shape {size, 1}
    std::vector<uint64_t> panels;
    std::vector<bool> enter;
    std::vector<uint64_t> flags;
    std::vector<int32_t> ids;
    std::vector<double> delpsi;
    std::vector<double> wavelength_cal;
    std::vector<uint64_t> experiment_ids;
    std::vector<std::string> identifiers;
#pragma endregion

    for (const std::string& expt_path : param_expt_paths) {
        // Get data from .expt file
        json data = json::parse(std::ifstream(expt_path));
        if (data.size() == 0) {
            logger.error("Experiment file " + expt_path + " is empty.\n");
            std::exit(1);
        }

        json experiment_list = data.at("experiment");

        for (int i_expt = 0; i_expt < experiment_list.size(); i_expt++) {
            // Compute the number of reflections predicted already (used when storing ids)
            const std::size_t num_reflections_initial = panels.size();

#pragma region Determine Experiment Parameters
            // FIXME: Extracting information from the json object manually here, as the ExperimentList
            // class does not exist yet.
            // In the future, please update the below logic.
            const json expt_details = experiment_list[i_expt];
            const std::string identifier = expt_details.at("identifier");

            // Obtain the indices indicating where experiment data can be found
            const std::size_t i_detector = expt_details.at("detector");
            const std::size_t i_goniometer = expt_details.at("goniometer");
            const std::size_t i_scan = expt_details.at("scan");
            const std::size_t i_crystal = expt_details.at("crystal");
            const std::size_t i_beam_data = expt_details.at("beam");
            const std::size_t i_imageset = expt_details.at("imageset");

            // Obtain json data components at the corresponding indices
            const json detector_data = data.at("detector")[i_detector];
            const json goniometer_data = data.at("goniometer")[i_goniometer];
            const json scan_data = data.at("scan")[i_scan];
            const json crystal_data = data.at("crystal")[i_crystal];
            const json beam_data = data.at("beam")[i_beam_data];
            const json imageset_data = data.at("imageset")[i_imageset];
            const std::string imageset_type = imageset_data.at("__id__");

            // Construct dx2 objects from the json data
            Detector detector(detector_data);
            Goniometer goniometer(goniometer_data);
            Scan scan(scan_data);
            // Note: Make crystal a shared_ptr to potentially add MosaicCrystalSauter2014 functionality in the future
            std::shared_ptr<Crystal> crystal;
            // if not MosaicCrystalSauter2014:
            crystal = std::make_shared<Crystal>(crystal_data);
            // if MosaicCrystalSauter2014:
            // crystal = std::make_shared<MosaicCrystalSauter2014>(crystal(crystal_data));
            gemmi::GroupOps crystal_symmetry_operations =
              crystal->get_space_group().operations();

            // Extract the A matrix
            const Matrix3d A = crystal->get_A_matrix();
#pragma endregion

#pragma region Determine Scan Parameters
            // Edit scan range and oscillation start. This adds 2 * param_buffer_size
            // images to the predictions.
            // FIXME: In most cases, this makes the program default to static prediction,
            // however the number of images after buffer adjustments may be such that the
            // condition for scan varying prediction is met. Explicitly set
            // param_force_static to true here?
            const int num_images =
              scan.get_image_range()[1] - scan.get_image_range()[0] + 1;
            const double osc0 = scan.get_oscillation()[0];
            if (param_buffer_size > 0) {
                scan = Scan({scan.get_image_range()[0] - param_buffer_size,
                             scan.get_image_range()[1] + param_buffer_size},
                            {scan.get_oscillation()[0]
                               - param_buffer_size * scan.get_oscillation()[1],
                             scan.get_oscillation()[1]});
                param_force_static = true;
            }
#pragma endregion

#pragma region Determine Beam Parameters
            BeamType beam_type;
            // Use the below for monochromatic and polychromatic prediction (where they represent the lower end of the wavelength)
            double wavelength;
            Vector3d s0;
            // Use the below for polychromatic prediction only
            double wavelength_poly_max = 0;
            if (beam_data.at("__id__") == "monochromatic") {
                MonochromaticBeam beam(beam_data);
                beam_type = BeamType::Monochromatic;
                wavelength = beam.get_wavelength();
                wavelength_poly_max = wavelength;
                s0 = beam.get_s0();
            } else if (beam_data.at("__id__") == "polychromatic") {
                PolychromaticBeam beam(beam_data);
                beam_type = BeamType::Polychromatic;
                wavelength = beam.get_wavelength_range()[0];
                wavelength_poly_max = beam.get_wavelength_range()[1];
                s0 = -beam.get_sample_to_source_direction() / wavelength;
            } else {
                logger.error(
                  "The beam's __id__ should be either monochromatic or polychromatic.");
                std::exit(1);
            }
#pragma endregion

#pragma region Determine dmin
            // Check if the minimum resolution paramenter (dmin) was passed in by the user,
            // if yes, check if it is a valid value; if not, assign a default.
            double dmin_min = 0.5 * wavelength;
            // FIXME: Need a better dmin_default from .expt file (like in DIALS)
            double dmin_default = dmin_min;
            if (!parser.is_used("dmin")) {
                param_dmin = dmin_default;
            } else if (param_dmin < dmin_min) {
                logger.error(
                  "Prediction at a dmin of {} is not possible with wavelength {}. "
                  "dmin "
                  "must be at least 0.5 times the wavelength.\nSetting dmin to the "
                  "default value of {}.\n",
                  param_dmin,
                  wavelength,
                  dmin_default);
                param_dmin = dmin_default;
            }
#pragma endregion

#pragma region Determine Prediction Parameters
            // Determine experiment type and extract data depending on type of scan
            PredictionType prediction_type{ExperimentType::Rotational,
                                           RotationalType::Static};
            ScanVaryingType sv_type;

            json s0_at_scan_points;
            json A_at_scan_points;
            json r_setting_at_scan_points;
            if (imageset_type != "ImageSequence") {
                prediction_type.experiment_type = ExperimentType::Stills;
                if (param_force_static)
                    logger.info(
                      "The experiment is not an ImageSequence. Ignoring the "
                      "--force_static "
                      "flag and falling back on stills prediction.");
            } else if (param_force_static) {
                prediction_type.experiment_type = ExperimentType::Rotational;
                prediction_type.rotational_type = RotationalType::Static;
            } else {
                if (beam_data.contains("s0_at_scan_points")) {
                    s0_at_scan_points = beam_data.at("s0_at_scan_points");
                    if (s0_at_scan_points.size() == num_images + 1) sv_type.beam = true;
                }
                // // Experimental feature: Support for scan-varying polychromatic beams.
                // // Uncomment the below code:
                // else if (beam_data.contains("unit_s0_at_scan_points")) {
                //     s0_at_scan_points = beam_data.at("unit_s0_at_scan_points");
                //     if (s0_at_scan_points.size() == num_images + 1) sv_type.beam = true;
                // }
                if (crystal_data.contains("A_at_scan_points")) {
                    A_at_scan_points = crystal_data.at("A_at_scan_points");
                    if (A_at_scan_points.size() == num_images + 1)
                        sv_type.crystal = true;
                }
                if (goniometer_data.contains("setting_rotation_at_scan_points")) {
                    r_setting_at_scan_points =
                      goniometer_data.at("setting_rotation_at_scan_points");
                    if (r_setting_at_scan_points.size() == num_images + 1)
                        sv_type.r_setting = true;
                }

                prediction_type.experiment_type = ExperimentType::Rotational;
                prediction_type.rotational_type =
                  (sv_type.beam || sv_type.crystal || sv_type.r_setting)
                    ? RotationalType::ScanVarying
                    : RotationalType::Static;
            }

            auto prediction_type_string = [&]() {
                if (prediction_type.experiment_type == ExperimentType::Stills)
                    return "stills";
                else {
                    // Rotational branch
                    if (prediction_type.rotational_type == RotationalType::ScanVarying)
                        return "scan-varying";
                    else
                        return "static";
                }
            };
            logger.info("{} {} prediction on {}",
                        (beam_type == BeamType::Monochromatic) ? "Monochromatic"
                                                               : "Polychromatic",
                        prediction_type_string(),
                        expt_path);
#pragma endregion

#pragma region Stills Prediction
            if (prediction_type.experiment_type == ExperimentType::Stills) {
                // A large enough angular tolerance allows plenty of Miller indices to be
                // available for checking against a finer tolerance.
                // Typical delta_psi tolerance is 0.0015, so a default of 0.005 is reasonable.
                // FIXME: Is there a clean way to determine this dynamically if the
                // mosaicity values are provided in the experiment file? (See below.)

                double angular_tolerance =
                  (beam_type == BeamType::Monochromatic) ? 0.005 : 0;
                Vector3d s0_lower = s0 * wavelength / wavelength_poly_max;
                // FIXME: This is very ugly because I had to make last-minute adjustments to accommodate
                // polychromatic prediction. Perhaps it is a better ideas to branch into mono and poly first,
                // then construct this generator! s0 here (for polychormatic) represents s0_upper (i.e. the
                // s0 corresponding to the lower wavelength)
                StillsIndexGenerator index_generator(
                  A, crystal_symmetry_operations, s0, s0_lower, angular_tolerance);

                for (;;) {
                    auto index = index_generator.next();
                    if (!index) break;

                    // Check if a reflection occurs at the given Miller index
                    // within the required resolution.
                    std::optional<Ray> ray;
                    if (beam_type == BeamType::Monochromatic) {
                        double delta_psi_tolerance = 0.0015;
                        if (crystal_data.find("mosaicity") != crystal_data.end()
                            && crystal_data.at("mosaicity") > 0) {
                            double ML_domain_size_ang =
                              crystal_data.at("ML_domain_size_ang");
                            double ML_half_mosaicity_deg =
                              crystal_data.at("ML_half_mosaicity_deg");
                            Vector3d h_vec = {(double)index.value()[0],
                                              (double)index.value()[1],
                                              (double)index.value()[2]};
                            double d = 1.0 / (A * h_vec).norm();
                            delta_psi_tolerance =
                              (d / ML_domain_size_ang)
                              + (ML_half_mosaicity_deg * M_PI / 360);
                        }
                        ray = predict_ray_monochromatic_stills(
                          index.value(), A, s0, param_dmin, delta_psi_tolerance);
                    } else
                        // AKA Laue prediction
                        ray = predict_ray_polychromatic_stills(index.value(),
                                                               A,
                                                               s0.normalized(),
                                                               wavelength,
                                                               wavelength_poly_max,
                                                               param_dmin);

                    if (!ray) continue;
                    // Append the ray
                    auto impact = detector.get_ray_intersection(ray->s1);
                    if (!impact) continue;

                    auto panel = impact->first;
                    auto coords_mm = impact->second;
                    auto coords_px =
                      detector.panels()[panel].mm_to_px(coords_mm[0], coords_mm[1]);

                    hkl.insert(
                      hkl.end(), std::begin(index.value()), std::end(index.value()));
                    enter.push_back(ray->entering);
                    s1.insert(s1.end(), std::begin(ray->s1), std::end(ray->s1));
                    xyz_mm.insert(
                      xyz_mm.end(), std::begin(coords_mm), std::end(coords_mm));
                    xyz_mm.push_back(0);
                    xyz_px.insert(
                      xyz_px.end(), std::begin(coords_px), std::end(coords_px));
                    xyz_px.push_back(0);
                    panels.push_back(panel);
                    flags.push_back(predicted_flag);
                    if (beam_type == BeamType::Monochromatic)
                        delpsi.push_back(ray->angle);
                    else {
                        wavelength_cal.push_back(1.0 / ray->s1.norm());
                        Vector3d s0_pred = s0.normalized() / ray->s1.norm();
                        s0_cal.insert(
                          s0_cal.end(), std::begin(s0_pred), std::end(s0_pred));
                    }
                }
            }
#pragma endregion

#pragma region Rotational Prediction
            else [[likely]] {
                // Rotational-experiment-specific code goes here

                const Vector3d m2 = goniometer.get_rotation_axis();
                // A Rotator object that generates rotations around axis m2
                const Rotator rotator(m2);
                const Matrix3d r_fixed = goniometer.get_sample_rotation();
                const Matrix3d r_setting = goniometer.get_setting_rotation();
                const double d_osc = scan.get_oscillation()[1];

                int z0 = scan.get_image_range()[0] - 1;
                int z1 = scan.get_image_range()[1];

                for (int frame = z0; frame < z1; frame++) {
                    int image_index = frame - z0;

                    // Define the potentially scan-varying vector (s0) and matrices (A and r_setting)
                    Vector3d s0_1 =
                      sv_type.beam ? vector_3d_from_json(s0_at_scan_points[image_index])
                                   : s0;
                    Vector3d s0_2 =
                      sv_type.beam
                        ? vector_3d_from_json(s0_at_scan_points[image_index + 1])
                        : s0;
                    Matrix3d A1 = sv_type.crystal
                                    ? matrix_3d_from_json(A_at_scan_points[image_index])
                                    : A;
                    Matrix3d A2 =
                      sv_type.crystal
                        ? matrix_3d_from_json(A_at_scan_points[image_index + 1])
                        : A;
                    Matrix3d r_setting_1 =
                      sv_type.r_setting
                        ? matrix_3d_from_json(r_setting_at_scan_points[image_index])
                        : r_setting;
                    Matrix3d r_setting_2 =
                      sv_type.r_setting
                        ? matrix_3d_from_json(r_setting_at_scan_points[image_index + 1])
                        : r_setting;

                    // Redefine A1 and A2 to encompass all 3 rotations
                    const double phi_beg = osc0 + image_index * d_osc;
                    const double phi_end = phi_beg + d_osc;
                    Matrix3d r_beg = rotator.rotation_matrix(phi_beg);
                    Matrix3d r_end = rotator.rotation_matrix(phi_end);
                    A1 = r_setting_1 * r_beg * r_fixed * A1;
                    A2 = r_setting_2 * r_end * r_fixed * A2;

                    ReekeIndexGenerator index_generator(
                      A1,
                      A2,
                      crystal_symmetry_operations,
                      s0_1,
                      s0_2,
                      param_dmin,
                      beam_type == BeamType::Monochromatic);

                    for (;;) {
                        auto index = index_generator.next();
                        if (!index) break;

                        // Check if a reflection occurs at the given Miller index
                        // within the required resolution.
                        std::array<std::optional<Ray>, 2> rays;
                        if (beam_type == BeamType::Monochromatic) {
                            if (prediction_type.rotational_type
                                == RotationalType::ScanVarying)
                                rays[0] = predict_ray_monochromatic_sv(index.value(),
                                                                       A1,
                                                                       A2,
                                                                       s0_1,
                                                                       s0_2,
                                                                       param_dmin,
                                                                       phi_beg,
                                                                       d_osc);
                            else {
                                // Monochromatic, Static, Rotation > 5 degrees.
                                // This is a new use case not supported by DIALS.
                                // For d_osc > 10 degrees, this exact calculator is much better
                                // than the DIALS-style approximate predictor
                                rays = predict_ray_monochromatic_static(
                                  index.value(),
                                  A1,
                                  r_setting_1,
                                  r_setting_1.inverse(),
                                  s0,
                                  m2,
                                  rotator,
                                  param_dmin,
                                  phi_beg,
                                  d_osc);
                            }

                        } else
                            // FIXME: Not implemented. This does not exist in DIALS: some new maths needs to be done.
                            rays[0] =
                              predict_ray_polychromatic_rotational(index.value(),
                                                                   A1,
                                                                   A2,
                                                                   s0_1.normalized(),
                                                                   s0_2.normalized(),
                                                                   wavelength,
                                                                   wavelength_poly_max,
                                                                   param_dmin,
                                                                   phi_beg,
                                                                   d_osc);

                        for (std::optional<Ray> ray : rays) {
                            if (!ray) continue;
                            // Append the ray
                            auto impact = detector.get_ray_intersection(ray->s1);
                            if (!impact) continue;

                            auto panel = impact->first;
                            auto coords_mm = impact->second;
                            auto coords_px = detector.panels()[panel].mm_to_px(
                              coords_mm[0], coords_mm[1]);

                            // Get the frame that a reflection with this angle will be observed at
                            double frame = z0 + (ray->angle - osc0) / d_osc;

                            hkl.insert(hkl.end(),
                                       std::begin(index.value()),
                                       std::end(index.value()));
                            enter.push_back(ray->entering);
                            s1.insert(s1.end(), std::begin(ray->s1), std::end(ray->s1));
                            xyz_mm.insert(
                              xyz_mm.end(), std::begin(coords_mm), std::end(coords_mm));
                            xyz_mm.push_back(ray->angle * M_PI / 180);
                            xyz_px.insert(
                              xyz_px.end(), std::begin(coords_px), std::end(coords_px));
                            xyz_px.push_back(frame);
                            panels.push_back(panel);
                            flags.push_back(predicted_flag);
                        }
                    }
                }
            }

            std::size_t num_new_reflections = panels.size() - num_reflections_initial;
            std::vector<int32_t> new_ids(num_new_reflections, i_expt);
            ids.insert(ids.end(), new_ids.begin(), new_ids.end());
            experiment_ids.push_back(i_expt);
            identifiers.push_back(identifier);
        }
    }
#pragma endregion

#pragma region Populate Reflection Table
    // Check if the vector sizes are consistent after prediction (before creating a ReflectionTable).
    // Note that delpsi, wavelength_cal, and s0_cal are allowed to either be 0 or equal to the row
    // size, depening on the type of prediction done.
    if ((hkl.size() != 3 * panels.size()) || (hkl.size() != 3 * enter.size())
        || (hkl.size() != s1.size()) || (hkl.size() != xyz_px.size())
        || (hkl.size() != xyz_mm.size()) || (hkl.size() != 3 * flags.size())
        || !(hkl.size() == 3 * delpsi.size() || delpsi.size() == 0)
        || !(hkl.size() == 3 * wavelength_cal.size() || wavelength_cal.size() == 0)
        || !(hkl.size() == s0_cal.size() || s0_cal.size() == 0)) {
        logger.error(
          "The sizes of the columns after prediction are not "
          "consistent with "
          "each other:\n hkl.size() = {}\n panel.size() = {}\n "
          "enter.size() "
          "= {}\n s1.size() = {}\n xyz_px.size() = {}\n xyz_mm.size() "
          "= {}\n flags.size() = {}\n ids.size() = {}\n delpsi.size() = "
          "{}\n wavelength_cal.size() = {}\n s0_cal.size() = {}\n",
          hkl.size(),
          panels.size(),
          enter.size(),
          s1.size(),
          xyz_px.size(),
          xyz_mm.size(),
          flags.size(),
          ids.size(),
          delpsi.size(),
          wavelength_cal.size(),
          s0_cal.size());
        std::exit(1);
    }

    // Store the size, once it has been verified as being consistent across columns.
    std::size_t sz = panels.size();

    predicted.add_column<int32_t>("miller_index", {sz, 3}, hkl);
    predicted.add_column<uint64_t>("panel", {sz}, panels);
    predicted.add_column<bool>("entering", {sz}, enter);
    predicted.add_column<double>("s1", {sz, 3}, s1);
    predicted.add_column<double>("xyzcal.px", {sz, 3}, xyz_px);
    predicted.add_column<double>("xyzcal.mm", {sz, 3}, xyz_mm);
    predicted.add_column<uint64_t>("flags", {sz}, flags);
    predicted.add_column<int32_t>("id", {sz}, ids);
    if (delpsi.size()) predicted.add_column<double>("delpsical.rad", {sz}, delpsi);
    if (wavelength_cal.size())
        predicted.add_column<double>("wavelength_cal", {sz}, wavelength_cal);
    if (s0_cal.size()) predicted.add_column<double>("s0_cal", {sz, 3}, s0_cal);
    predicted.set_experiment_ids(experiment_ids);
    predicted.set_identifiers(identifiers);

#pragma endregion

#pragma region Dynamic Shadowing
    // If not ignoring shadows, look for reflections in the masked region
    if (param_dynamic_shadows) {
    }

    /*
	if not params.ignore_shadows:
			try:
					experiments = ExperimentListFactory.from_json(
							experiments.as_json(), check_format=True
					)
			except OSError as e:
					sys.exit(
							f"Unable to read image data. Please check {e.filename} is accessible"
					)
			shadowed = filter_shadowed_reflections(
					experiments, predicted_all, experiment_goniometer=True
			)
			predicted_all = predicted_all.select(~shadowed)
	*/
#pragma endregion

    // FIXME: DIALS tries to find bounding boxes for each experiment.
    // If this is important, implement this. If not, leave this commented out.
    /*
	try:
			predicted_all.compute_bbox(experiments)
	except Exception:
			pass
	*/

#pragma region Write to File
    // Save reflections to file
    predicted.write(output_file_path);

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    logger.info("Saved {} reflections to {}.", sz, output_file_path);
    logger.info("Total time for prediction: {:.4f}s", elapsed_time.count());
#pragma endregion
    return 0;
}