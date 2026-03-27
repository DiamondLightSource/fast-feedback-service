
#include <gtest/gtest.h>
#include "h5read.h"


#ifndef THAUMATIN_DATA_PATH
#error "THAUMATIN_DATA_PATH not set"
#endif


TEST(H5read, test_h5read){
    std::string filename = std::string(THAUMATIN_DATA_PATH) + "/thau_2_1.nxs";
    auto reader = H5Read(filename);
    auto [image_slow, image_fast] = reader.image_shape();
    auto mask = reader.get_mask().value();

    std::unordered_map<int, int> total_counts_by_image;

    for (size_t j = 0; j < 3; j++) {
        if (reader.is_image_available(j)){
            auto image = reader.get_image(j);
            int total = 0;
            for (int index = 0; index<image_fast*image_slow; index++){
                if (mask[index] != 0){
                    total += image.data.data()[index];
                }
            }
            total_counts_by_image[j] = total;
        }
    }
    EXPECT_EQ(total_counts_by_image[0], 504245);
    EXPECT_EQ(total_counts_by_image[1], 503554);
    EXPECT_EQ(total_counts_by_image[2], 506110);
}


