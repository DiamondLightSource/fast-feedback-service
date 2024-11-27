#ifndef REFLECTION_DATA
#define REFLECTION_DATA
#include <vector>
#include <Eigen/Dense>
using Eigen::Vector3d;
using Eigen::Vector3i;

struct reflection_data {
    std::vector<std::size_t> flags;
    std::vector<Vector3d> xyzobs_mm;
    std::vector<Vector3d> xyzcal_mm;
    std::vector<Vector3d> s1;
    std::vector<Vector3i> miller_indices;
    std::vector<bool> entering;
};

reflection_data select(reflection_data data, std::vector<bool> sel){
    reflection_data selected;
    for (int i=0;i<sel.size();i++){
        if (sel[i]){
            selected.flags.push_back(data.flags[i]);
            selected.xyzobs_mm.push_back(data.xyzobs_mm[i]);
            selected.xyzcal_mm.push_back(data.xyzcal_mm[i]);
            selected.s1.push_back(data.s1[i]);
            selected.miller_indices.push_back(data.miller_indices[i]);
            selected.entering.push_back(data.entering[i]);
        }
    }
    return selected;
}

reflection_data select(reflection_data data, std::vector<std::size_t> sel){
    reflection_data selected;
    for (int i=0;i<sel.size();i++){
        int index = sel[i];
        selected.flags.push_back(data.flags[index]);
        selected.xyzobs_mm.push_back(data.xyzobs_mm[index]);
        selected.xyzcal_mm.push_back(data.xyzcal_mm[index]);
        selected.s1.push_back(data.s1[index]);
        selected.miller_indices.push_back(data.miller_indices[index]);
        selected.entering.push_back(data.entering[index]);
    }
    return selected;
}

#endif //REFLECTION_DATA