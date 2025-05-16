#include <dx2/crystal.hpp>
#include <dx2/utils.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "gemmi/symmetry.hpp"
#include "gemmi/unitcell.hpp"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

// work in degrees
constexpr double half_pi = 90.0;
constexpr double min_angle = 20.0;

// A class to determine candadite orientation matrices by combining potential lattice vectors.

class CandidateOrientationMatrices {
  public:
    CandidateOrientationMatrices(const std::vector<Vector3d>& basis_vectors,
                                 int max_combinations = -1)
        : max_combinations(max_combinations), index(0) {
        n = basis_vectors.size();
        n = std::min(n, 100);
        // Calculate combinations of lattice vectors
        truncated_basis_vectors = {basis_vectors.begin(), basis_vectors.begin() + n};
        combinations.reserve(n * n * n / 4);  // could work this out exactly...
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                for (int k = j + 1; k < n; k++) {
                    combinations.push_back({i, j, k});
                }
            }
        }
        // Sort them from lowest to highest magnitude.
        std::stable_sort(
          combinations.begin(), combinations.end(), [](Vector3i a, Vector3i b) {
              return a.squaredNorm()
                     < b.squaredNorm();  // note can't use norm as get int truncation after std::sqrt.
          });
        // Truncate to the minimum of the number of possible combinations or the requested limit.
        int extent = std::min(max_combinations, static_cast<int>(combinations.size()));
        truncated_combinations = {combinations.begin(), combinations.begin() + extent};
    }

    bool has_next() {
        return index < truncated_combinations.size();
    }

    // Generate the next valid combination that meets a set of criteria.
    std::optional<Crystal> next() {
        while (index < truncated_combinations.size()) {
            Vector3i comb = truncated_combinations[index];
            Vector3d v1 = truncated_basis_vectors[comb[0]];
            Vector3d v2 = truncated_basis_vectors[comb[1]];
            index++;
            double gamma = angle_between_vectors_degrees(v1, v2);
            if (gamma < min_angle || (180 - gamma) < min_angle) {
                continue;
            }
            Vector3d crossprod = v1.cross(v2);
            if (gamma < half_pi) {
                v2 = -v2;
                crossprod = -crossprod;
            }
            Vector3d v3 = truncated_basis_vectors[comb[2]];
            if (std::abs(half_pi - angle_between_vectors_degrees(crossprod, v3))
                < min_angle) {
                continue;
            }
            double alpha = angle_between_vectors_degrees(v2, v3);
            if (alpha < half_pi) {
                v3 = -v3;
            }
            if (crossprod.dot(v3) < 0) {
                v1 = -v1;
                v2 = -v2;
                v3 = -v3;
            }
            gemmi::SpaceGroup space_group = *gemmi::find_spacegroup_by_name("P1");
            Crystal c{v1, v2, v3, space_group};
            c.niggli_reduce();
            gemmi::UnitCell cell = c.get_unit_cell();
            if (cell.volume > (cell.a * cell.b * cell.c / 100.0)) {
                return c;
            }
        }
        return {};
    }

  private:
    std::vector<Vector3d> truncated_basis_vectors{};
    std::vector<Vector3i> combinations{};
    std::vector<Vector3i> truncated_combinations{};
    int n;
    int max_combinations;
    size_t index;
};