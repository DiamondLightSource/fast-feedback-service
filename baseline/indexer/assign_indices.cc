#ifndef ASSIGN_INDICES_H
#define ASSIGN_INDICES_H
#include <Eigen/Dense>
#include <cmath>
#include <experimental/mdspan>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

template <typename T>
using mdspan_type = std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;


constexpr double pi_4 = M_PI / 4;

/**
 * @brief Assigns miller indices to reciprocal lattice points.
 * @param A The crystal A-matrix.
 * @param rlp The vector of reciprocal lattice points.
 * @param xyzobs_mm The vector of observed xyz positions, in mm.
 * @param tolerance The tolerance within which the fractional miller index must be for acceptance.
 * @returns A pair containing the assigned miller indices and the number of reciprocal lattice points successfully indexed.
 */
std::pair<std::vector<int>, int> assign_indices_global(
  Matrix3d const &A,
  mdspan_type<double> const &rlp,
  mdspan_type<double> const &xyzobs_mm,
  const double tolerance = 0.3) {
    // Consider only a single lattice.
    std::vector<int> miller_indices_data(rlp.size());
    mdspan_type<int> miller_indices(miller_indices_data.data(), rlp.extent(0), 3);
    std::vector<int> crystal_ids(rlp.extent(0));
    std::vector<double> lsq_vector(rlp.extent(0));
    Vector3i miller_index_zero{{0, 0, 0}};
    typedef std::multimap<
      Vector3i,
      std::size_t,
      std::function<bool(const Eigen::Vector3i &, const Eigen::Vector3i &)> >
      hklmap;

    hklmap miller_idx_to_iref([](const Vector3i &a, const Vector3i &b) -> bool {
        return std::lexicographical_compare(
          a.data(), a.data() + a.size(), b.data(), b.data() + b.size());
    });
    // Iterate through the data, assigning a miller index if within the tolerance.
    const Matrix3d A_inv = A.inverse();
    const double tolsq = tolerance * tolerance;
    int count = 0;
    for (int i = 0; i < rlp.extent(0); ++i) {
        Eigen::Map<Vector3d> rlp_this(&rlp(i,0));
        Vector3d hkl_f = A_inv * rlp_this;
        for (std::size_t j = 0; j < 3; j++) {
            miller_indices(i,j) = static_cast<int>(round(hkl_f[j]));
        }
        Vector3d diff{{0, 0, 0}};
        diff[0] = static_cast<double>(miller_indices(i,0)) - hkl_f[0];
        diff[1] = static_cast<double>(miller_indices(i,1)) - hkl_f[1];
        diff[2] = static_cast<double>(miller_indices(i,2)) - hkl_f[2];
        double l_sq = diff.squaredNorm();
        if (l_sq > tolsq) {
            miller_indices(i,0) = 0;
            miller_indices(i,1) = 0;
            miller_indices(i,2) = 0;
            crystal_ids[i] = -1;
        } else if (miller_indices(i,0) == 0 && miller_indices(i,1) == 0 && miller_indices(i,2) == 0) {
            crystal_ids[i] = -1;
        } else {
            Vector3i midx = {miller_indices(i,0), miller_indices(i,1), miller_indices(i,2)};
            miller_idx_to_iref.emplace(midx, i);
            lsq_vector[i] = l_sq;
            count++;
        }
    }
    // if more than one spot can be assigned the same miller index then
    // choose the closest one
    Vector3i curr_hkl{{0, 0, 0}};
    std::vector<std::size_t> i_same_hkl;
    for (hklmap::iterator it = miller_idx_to_iref.begin();
         it != miller_idx_to_iref.end();
         it++) {
        if (it->first != curr_hkl) {
            if (i_same_hkl.size() > 1) {
                for (int i = 0; i < i_same_hkl.size(); i++) {
                    const std::size_t i_ref = i_same_hkl[i];
                    for (int j = i + 1; j < i_same_hkl.size(); j++) {
                        const std::size_t j_ref = i_same_hkl[j];
                        if (crystal_ids[i_ref] == -1) {
                            continue;
                        }
                        if (crystal_ids[j_ref] == -1) {
                            continue;
                        }
                        double phi_i = xyzobs_mm(i_ref, 2);
                        double phi_j = xyzobs_mm(j_ref, 2);
                        if (std::abs(phi_i - phi_j) > pi_4) {
                            continue;
                        }
                        if (lsq_vector[j_ref] < lsq_vector[i_ref]) {
                            miller_indices(i_ref,0) = 0;
                            miller_indices(i_ref,1) = 0;
                            miller_indices(i_ref,2) = 0;
                            crystal_ids[i_ref] = -1;
                            count--;
                        } else {
                            miller_indices(j_ref,0) = 0;
                            miller_indices(j_ref,1) = 0;
                            miller_indices(j_ref,2) = 0;
                            crystal_ids[j_ref] = -1;
                            count--;
                        }
                    }
                }
            }
            curr_hkl = it->first;
            i_same_hkl.clear();
        }
        i_same_hkl.push_back(it->second);
    }

    // Now do the final group!
    if (i_same_hkl.size() > 1) {
        for (int i = 0; i < i_same_hkl.size(); i++) {
            const std::size_t i_ref = i_same_hkl[i];
            for (int j = i + 1; j < i_same_hkl.size(); j++) {
                const std::size_t j_ref = i_same_hkl[j];
                if (crystal_ids[i_ref] == -1) {
                    continue;
                }
                if (crystal_ids[j_ref] == -1) {
                    continue;
                }
                double phi_i = xyzobs_mm(i_ref, 2);
                double phi_j = xyzobs_mm(j_ref, 2);
                if (std::abs(phi_i - phi_j) > pi_4) {
                    continue;
                }
                if (lsq_vector[j_ref] < lsq_vector[i_ref]) {
                    miller_indices(i_ref, 0) = 0;
                    miller_indices(i_ref, 1) = 0;
                    miller_indices(i_ref, 2) = 0;
                    crystal_ids[i_ref] = -1;
                    count--;
                } else {
                    miller_indices(j_ref, 0) = 0;
                    miller_indices(j_ref, 1) = 0;
                    miller_indices(j_ref, 2) = 0;
                    crystal_ids[j_ref] = -1;
                    count--;
                }
            }
        }
    }

    return {miller_indices_data, count};
}

#endif