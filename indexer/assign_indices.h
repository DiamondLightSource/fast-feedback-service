#include <math.h>
#include <Eigen/Dense>
#include <iostream>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

std::vector<Vector3i> assign_indices_global(Matrix3d A, std::vector<Vector3d> rlp, std::vector<double> phi, double tolerance = 0.3){
    // Consider only a single lattice.
    std::vector<Vector3i> miller_indices(rlp.size());
    std::vector<int> crystal_ids(rlp.size());
    std::vector<double> lsq_vector(rlp.size());

    // map of milleridx to
    typedef std::multimap<Vector3i, std::size_t, std::function<bool(const Eigen::Vector3i&,const Eigen::Vector3i&)> > hklmap;
    
    hklmap miller_idx_to_iref([](const Vector3i & a, const Vector3i & b)->bool
    {
        return std::lexicographical_compare(
        a.data(),a.data()+a.size(),
        b.data(),b.data()+b.size());
    });
    
    //hklmap miller_idx_to_iref;
    double pi_4 = M_PI / 4;
    Vector3i miller_index_zero{{0, 0, 0}};
    Matrix3d A_inv = A.inverse();
    //std::cout << A_inv << std::endl;
    double tolsq = std::pow(tolerance, 2);
    for (int i=0;i<rlp.size();i++){
        Vector3d rlp_this = rlp[i];
        Vector3d hkl_f = A_inv * rlp_this;
        for (std::size_t j = 0; j < 3; j++) {
            miller_indices[i][j] = (int)round(hkl_f[j]);
        }
        Vector3d diff{{0,0,0}};
        diff[0] = (double)miller_indices[i][0] - hkl_f[0];
        diff[1] = (double)miller_indices[i][1] - hkl_f[1];
        diff[2] = (double)miller_indices[i][2] - hkl_f[2];
        double l_sq = diff.squaredNorm();
        /*if (i == 693){
            std::cout << l_sq << std::endl;
            std::cout << miller_indices[i][0] << " " << miller_indices[i][1] << " " << miller_indices[i][2] << std::endl;
            std::cout << hkl_f[0] << " " << hkl_f[1] << " " << hkl_f[2] << std::endl;
            std::cout << rlp_this[0] << " " << rlp_this[1] << " " << rlp_this[2] << std::endl;
        }*/
        if (l_sq > tolsq){
            miller_indices[i] = {0,0,0};
            crystal_ids[i] = -1;
        }
        else if (miller_indices[i] == miller_index_zero){
            crystal_ids[i] = -1;
        }
        else {
            miller_idx_to_iref.insert({miller_indices[i],i});
            lsq_vector[i] = l_sq;
        }
    }
    // if more than one spot can be assigned the same miller index then
    // choose the closest one
    /*for (hklmap::iterator it = miller_idx_to_iref.begin(); it != miller_idx_to_iref.end(); it++){
        std::cout << it->first[0] << " " << it->first[1] << " " << it->first[2] << " " << it->second << std::endl;
    }*/

    Vector3i curr_hkl{{0, 0, 0}};
    std::vector<std::size_t> i_same_hkl;
    for (hklmap::iterator it = miller_idx_to_iref.begin(); it != miller_idx_to_iref.end(); it++){
        if (it->first != curr_hkl){
            if (i_same_hkl.size() > 1) {
                for (int i = 0; i < i_same_hkl.size(); i++) {
                    const std::size_t i_ref = i_same_hkl[i];
                    for (int j = i + 1; j < i_same_hkl.size(); j++) {
                        const std::size_t j_ref = i_same_hkl[j];
                        double phi_i = phi[i_ref];
                        double phi_j = phi[j_ref];
                        if (std::abs(phi_i - phi_j) > pi_4) {
                            continue;
                        }
                        if (lsq_vector[i_ref] < lsq_vector[j_ref]){
                            miller_indices[i_ref] = {0,0,0};
                            crystal_ids[i_ref] = -1;
                        }
                        else {
                            miller_indices[j_ref] = {0,0,0};
                            crystal_ids[j_ref] = -1;
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
                double phi_i = phi[i_ref];
                double phi_j = phi[j_ref];
                if (std::abs(phi_i - phi_j) > pi_4) {
                    continue;
                }
                if (lsq_vector[i_ref] < lsq_vector[j_ref]){
                    miller_indices[i_ref] = {0,0,0};
                    crystal_ids[i_ref] = -1;
                }
                else {
                    miller_indices[j_ref] = {0,0,0};
                    crystal_ids[j_ref] = -1;
                }
            }
        }
    }


    return miller_indices;
}