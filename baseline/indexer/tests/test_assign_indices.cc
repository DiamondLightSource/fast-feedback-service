#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "assign_indices.cc"

using Eigen::Vector3d;

TEST(BaselineIndexer, assign_indices_test) {
    // Reflections (22,-2,-1) x2, (22,-4,-4) x2 and (22,-4,5) from beta lactamase example dataset (c2sum)
    // These will loop through in the order (22,-4,4), (22,-4,5), (22,-2,-1).
    Matrix3d A{{-0.0134, -0.0227, -0.0009},
                {-0.0053,  0.0030, -0.0140},
                { 0.0203, -0.0098, -0.0036}};
    std::vector<Vector3d> rlp;
    std::vector<Vector3d> xyzobs_mm;
    rlp.push_back({-0.20806554291174043, -0.20006695964877577, 0.46900930253991086});
    rlp.push_back({-0.20711023322729, -0.1844213770514764, 0.47299170443928473});
    rlp.push_back({-0.24784778679168532, -0.10882416182889978, 0.47089025552307});
    rlp.push_back({-0.20693750699327926, -0.1837374215284657, 0.4732856446956432});
    rlp.push_back({-0.24768402495471797, -0.11030902243741092, 0.4706897967841226});
    // Add a final point which will be outside a tolerance of 0.2
    rlp.push_back({-0.565, 0.15, 0.027});
    xyzobs_mm.push_back({128.09295901467863, 102.49866279545705, 2.325651228282444});
    xyzobs_mm.push_back({128.5569574846125, 103.27824528734998, 2.360557813322331});
    xyzobs_mm.push_back({119.93326007874127, 113.97862045047734, 2.5612706773016787});
    xyzobs_mm.push_back({128.71319842438626, 317.368360910889, 4.620759194654988});
    xyzobs_mm.push_back({119.97077430898872, 306.5586092785435, 4.751658888554562});
    xyzobs_mm.push_back({65.0, 226.0, 0.013});
    int count;
    std::vector<Vector3i> miller_index;
    // Use a tolerance of 0.2 to filter out the last point.
    std::tie(miller_index, count) = assign_indices_global(A, rlp, xyzobs_mm, 0.2);
    std::vector<Vector3i> expected_miller_indices{{22, -4, 5}, {22, -4, 4}, {22, -2, -1}, {22, -4, 4}, {22, -2, -1}};
    EXPECT_EQ(count, 5);
    for (int i=0;i<expected_miller_indices.size();++i){
        EXPECT_EQ(miller_index[i], expected_miller_indices[i]);
    }
}   