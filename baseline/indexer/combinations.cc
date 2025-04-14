#include <vector>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <dx2/crystal.h>
#include "gemmi/unitcell.hpp"
#include "gemmi/symmetry.hpp"
#include <dx2/utils.h>

using Eigen::Vector3d;
using Eigen::Vector3i;
using Eigen::Matrix3d;

bool compare_comb(Vector3i a, Vector3i b){
    return a.squaredNorm() < b.squaredNorm(); // note can't use norm as get int truncation after std::sqrt.
}

class CandidateOrientationMatrices {
public:
    CandidateOrientationMatrices(
        const std::vector<Vector3d>& basis_vectors, int max_combinations = -1)
        :  max_combinations(max_combinations), index(0) {
            n = basis_vectors.size();
            n = std::min(n, 100);
            truncated_basis_vectors = {basis_vectors.begin(), basis_vectors.begin() + n};
            // now get combinations of
            combinations.reserve(n*n*n/4); // could work this out exactly...
            norms.reserve(n*n*n/4);
            for (int i=0;i<n;i++){
                for (int j=i+1;j<n;j++){
                    for (int k=j+1;k<n;k++){
                        combinations.push_back({i,j,k});
                        norms.push_back(std::pow(i*i + j*j + k*k, 0.5));
                    }
                }
            }
            std::stable_sort(combinations.begin(), combinations.end(), compare_comb);
            truncated_combinations = {combinations.begin(), combinations.begin()+max_combinations};
            // work in degrees for now
            half_pi = 90.0;//0.5 * M_PI;
            min_angle = 20.0;//M_PI * 20.0 / 180.0;
        }

    bool has_next() {
        return index < truncated_combinations.size();
    }

    Crystal next() {
        while (index < truncated_combinations.size()){
            Vector3i comb = truncated_combinations[index];
            Vector3d v1 = truncated_basis_vectors[comb[0]];
            Vector3d v2 = truncated_basis_vectors[comb[1]];
            index++;
            double gamma = angle_between_vectors_degrees(v1,v2);
            if (gamma < min_angle || (180 - gamma) < min_angle) {
                //std::cout << "skipping comb " << comb[0] << " " << comb[1] << " " << comb[2] << std::endl;
                continue;
            }
            Vector3d crossprod = v1.cross(v2);
            if (gamma < half_pi){
                v2 = -v2;
                crossprod = -crossprod;
            }
            Vector3d v3 = truncated_basis_vectors[comb[2]];
            if (std::abs(half_pi - angle_between_vectors_degrees(crossprod, v3)) < min_angle){
                //std::cout << "skipping comb " << comb[0] << " " << comb[1] << " " << comb[2] << std::endl;
                continue;
            }
            double alpha = angle_between_vectors_degrees(v2, v3);
            if (alpha < half_pi){
                v3 = -v3;
            }
            if (crossprod.dot(v3) < 0){
                v1 = -v1;
                v2 = -v2;
                v3 = -v3;
            }
            //std::cout << "returning comb " << comb[0] << " " << comb[1] << " " << comb[2] << std::endl;
            gemmi::SpaceGroup space_group = *gemmi::find_spacegroup_by_name("P1");
            Crystal c{v1,v2,v3, space_group};
            c.niggli_reduce();
            gemmi::UnitCell cell = c.get_unit_cell();
            if (cell.volume > (cell.a * cell.b * cell.c / 100.0)){
                std::cout << "Returning combination: " << comb[0] << "," << comb[1] << "," << comb[2] << std::endl;
                return c;
            }
                
        }
        throw std::out_of_range("No more combinations available");
    }

private:
    std::vector<Vector3d> truncated_basis_vectors{};
    std::vector<Vector3i> combinations{};
    std::vector<Vector3i> truncated_combinations{};
    std::vector<int> norms{};
    //flex::vec3_double::parts_type parts;
    //flex::int_array i, j, k;
    int n;
    int max_combinations;
    size_t index;
    double half_pi;
    double min_angle;
};