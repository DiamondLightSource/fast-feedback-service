#include <Eigen/Dense>
#include <gemmi/symmetry.hpp>
#include <optional>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;

#pragma region Index Generators

/**
 * A class to generate miller indices for rotational experiments using the Reeke algorithm.
 */
class ReekeIndexGenerator {
  public:
    ReekeIndexGenerator(const Matrix3d &A1,
                        const Matrix3d &A2,
                        gemmi::GroupOps &crystal_symmetry_operations,
                        const Vector3d &s0_1,
                        const Vector3d &s0_2,
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
    auto calc_h_limits_resolution(const Vector3d &a, const Vector3d &s0)
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
    auto calc_k_limits_ewald(const Matrix4d &T, const int h)
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
    auto calc_k_limits_resolution(const Matrix4d &T, const int h)
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
    auto calc_l_limits_ewald(const Matrix4d &T, const int h, const int k)
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
    auto calc_l_limits_resolution(const Matrix4d &T, const int h, const int k)
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
    StillsIndexGenerator(const Matrix3d &A,
                         gemmi::GroupOps &crystal_symmetry_operations,
                         const Vector3d &s0_upper,
                         const Vector3d &s0_lower,
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