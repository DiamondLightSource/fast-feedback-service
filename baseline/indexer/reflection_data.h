#ifndef REFLECTION_DATA
#define REFLECTION_DATA
#include <vector>
#include <Eigen/Dense>
using Eigen::Vector3d;
using Eigen::Vector3i;

struct reflection_data {
    std::vector<std::size_t> flags; // needed?
    std::vector<Vector3d> xyzobs_mm;
    std::vector<Vector3d> xyzcal_mm; // starts empty
    std::vector<Vector3d> s1;
    std::vector<Vector3d> rlp;
    std::vector<Vector3i> miller_indices; // starts empty
    std::vector<bool> entering;
};

reflection_data select(reflection_data const& data, std::vector<bool> const& sel){
    reflection_data selected;
    bool has_miller = (data.miller_indices.size() > 0);
    bool has_xyzcal = (data.xyzcal_mm.size() > 0);
    for (int i=0;i<sel.size();i++){
        if (sel[i]){
            selected.flags.push_back(data.flags[i]);
            selected.xyzobs_mm.push_back(data.xyzobs_mm[i]);
            selected.rlp.push_back(data.rlp[i]);
            if (has_xyzcal){
                selected.xyzcal_mm.push_back(data.xyzcal_mm[i]);
            }
            selected.s1.push_back(data.s1[i]);
            if (has_miller){
                selected.miller_indices.push_back(data.miller_indices[i]);
            }
            selected.entering.push_back(data.entering[i]);
        }
    }
    return selected;
}

reflection_data select(reflection_data const& data, std::vector<std::size_t> const& sel){
    reflection_data selected;
    for (int i=0;i<sel.size();i++){
        int index = sel[i];
        selected.flags.push_back(data.flags[index]);
        selected.xyzobs_mm.push_back(data.xyzobs_mm[index]);
        selected.xyzcal_mm.push_back(data.xyzcal_mm[index]);
        selected.s1.push_back(data.s1[index]);
        selected.rlp.push_back(data.rlp[index]);
        selected.miller_indices.push_back(data.miller_indices[index]);
        selected.entering.push_back(data.entering[index]);
    }
    return selected;
}

#endif //REFLECTION_DATA