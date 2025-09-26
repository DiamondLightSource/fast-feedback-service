#include <Eigen/Dense>
#include <dx2/crystal.hpp>
#include <vector>

#include "assign_indices.cc"
#include "ffs_logger.hpp"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;
template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;

// Following the code in dials/algorithms/indexing/non_primivite_basis.py

inline int mod_positive(int x, int y) {
    x %= y;
    if (x < 0) {
        x += y;
    }
    return x;
}

std::vector<int> absence_test(const std::vector<Vector3i> &hkl,
                              const int &mod,
                              Vector3i vecrep) {
    std::vector<int> cumulative(mod, 0);
    for (const Vector3i &millerindex : hkl) {
        int pattern_sum = millerindex.dot(vecrep);
        cumulative[mod_positive(pattern_sum, mod)]++;
    }
    return cumulative;
}

struct reindex_transforms {
    int modularity;
    Vector3i vector;
    Matrix3d transformation;
};

// This is always the same, ideally would make a constexpr but issues with Eigen objects not being literal types
std::vector<reindex_transforms> generate_reindex_transformations() {
    const std::vector<int> modularities = {2, 3, 5};
    // generate combinations
    std::vector<Vector3i> points;
    points.reserve(11 * 11 * 11);
    for (int i = 5; i > -6; --i) {
        for (int j = 5; j > -6; --j) {
            for (int k = 5; k > -6; --k) {
                points.push_back({i, j, k});
            }
        }
    }
    std::sort(points.begin(), points.end(), [](const Vector3i a, const Vector3i b) {
        int d1 = a.dot(a);
        int d2 = b.dot(b);
        if (d1 == d2) {
            if (a.sum() == b.sum()) {
                return !std::lexicographical_compare(
                  a.data(),
                  a.data() + a.size(),
                  b.data(),
                  b.data() + b.size());  // (1,0,0) before (0,1,0), before (0,0,1).
            }
            return a.sum()
                   > b.sum();  // Secondary criterion - points with higher sum prioritised
        }
        return d1 < d2;  // Primary criterion - return closest to origin
    });
    // remove the first element (0,0,0)
    points.erase(points.begin());

    Vector3i zero = {0, 0, 0};
    std::vector<Vector3i> representatives;
    for (const Vector3i &point : points) {
        if (point.dot(point) > 6) {
            break;
        }
        // see if collinear with any existing;
        bool is_collinear = false;
        for (const Vector3i &repr : representatives) {
            if (point.cross(repr) == zero) {
                is_collinear = true;
                break;
            }
        }
        if (!is_collinear) {
            representatives.push_back(point);
        }
    }

    // Now generate reindex matrices
    std::vector<reindex_transforms> reindex;
    for (const Vector3i &repr : representatives) {
        for (const int &modularity : modularities) {
            std::vector<Vector3i> candidate_points;
            for (const Vector3i &point : points) {
                if ((point.dot(repr) % modularity) == 0) {
                    candidate_points.push_back(point);
                }
            }
            Vector3i first = candidate_points.front();
            Vector3i second;
            Vector3i third;
            candidate_points.erase(candidate_points.begin());
            while (true) {
                second = candidate_points.front();
                candidate_points.erase(candidate_points.begin());
                if (!(second.cross(first) == zero)) {
                    break;
                }
            }
            while (true) {
                third = candidate_points.front();
                candidate_points.erase(candidate_points.begin());
                if (!(second.cross(first).dot(third) == 0)) {
                    break;
                }
            }
            Matrix3d A{{static_cast<double>(first[0]),
                        static_cast<double>(first[1]),
                        static_cast<double>(first[2])},
                       {static_cast<double>(second[0]),
                        static_cast<double>(second[1]),
                        static_cast<double>(second[2])},
                       {static_cast<double>(third[0]),
                        static_cast<double>(third[1]),
                        static_cast<double>(third[2])}};
            if (A.determinant() < 0) {
                A = Matrix3d{{static_cast<double>(second[0]),
                              static_cast<double>(second[1]),
                              static_cast<double>(second[2])},
                             {static_cast<double>(first[0]),
                              static_cast<double>(first[1]),
                              static_cast<double>(first[2])},
                             {static_cast<double>(third[0]),
                              static_cast<double>(third[1]),
                              static_cast<double>(third[2])}};
            }
            reindex_transforms r{modularity, repr, A};
            reindex.push_back(r);
        }
    }
    return reindex;
}

std::vector<reindex_transforms> transforms = generate_reindex_transformations();

Matrix3d null{};

/**
 * @brief Perform absence tests to evaluate potential systematic absences.
 * @param hkl A vector of miller indices.
 * @param threshold A threshold for positive identification of an absence.
 * @returns A transformation matrix to reindex and remove the absence.
 */
Matrix3d detect(const std::vector<Vector3i> &hkl, double threshold = 0.9) {
    for (const reindex_transforms &transform : transforms) {
        std::vector<int> cumulative =
          absence_test(hkl, transform.modularity, transform.vector);
        for (int i = 0; i < transform.modularity; ++i) {
            if ((static_cast<double>(cumulative[i]) / hkl.size()) > threshold
                && i == 0) {
                logger.debug(
                  "Detected exclusive presence of {}H {}K {}L = {}N, remainder {}",
                  transform.vector[0],
                  transform.vector[1],
                  transform.vector[2],
                  transform.modularity,
                  i);
                return transform.transformation;
            }
        }
    }
    return null;
}

/**
 * @brief Correct the miller indices and crystal cell for non-primitive basis set choices.
 * @param hkl The miller indices.
 * @param crystal The crystal model.
 * @param rlp The vector of reciprocal lattice points.
 * @param xyzobs_mm The vector of observed xyz positions, in mm.
 * @param threshold A threshold for positive identification of an absence.
 * @returns The number of rlps that are indexed after application of this function.
 */
int correct(std::vector<int> &hkl,
            Crystal &crystal,
            mdspan_type<double> const &rlp,
            mdspan_type<double> const &xyzobs_mm,
            double threshold = 0.9) {
    Vector3i null_miller = {0, 0, 0};
    int count;  // num indexed
    while (true) {
        mdspan_type<int> miller_indices(hkl.data(), hkl.size() / 3, 3);
        std::vector<Vector3i> selected_miller;
        selected_miller.reserve(hkl.size() / 3);
        for (int i = 0; i < miller_indices.extent(0); ++i) {
            Vector3i midx = {
              miller_indices(i, 0), miller_indices(i, 1), miller_indices(i, 2)};
            if (midx != null_miller) {
                selected_miller.push_back(midx);
            }
        }
        count = selected_miller.size();
        if (selected_miller.size() == 0) {
            break;
        }
        Matrix3d T = detect(selected_miller, threshold);
        if (T == null) {
            break;
        }
        Matrix3d direct_matrix = crystal.get_A_matrix().inverse();
        Matrix3d M = T.inverse().transpose();
        Matrix3d new_direct = M * direct_matrix;
        crystal.set_A_matrix(new_direct.inverse());
        crystal.niggli_reduce();
        assign_indices_results results =
          assign_indices_global(crystal.get_A_matrix(), rlp, xyzobs_mm);
        hkl = results.miller_indices_data;
        count = results.number_indexed;
    }
    return count;
}
