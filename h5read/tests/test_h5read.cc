
#include <gtest/gtest.h>
#include "h5read.h"


#ifndef THAUMATIN_DATA_PATH
#error "THAUMATIN_DATA_PATH not set"
#endif


TEST(H5read, test_h5read){
    std::string filename = std::string(THAUMATIN_DATA_PATH) + "/thau_2_1.nxs";
    h5read_handle *obj = h5read_open(filename.c_str());
    size_t n_images = h5read_get_number_of_images(obj);
    uint16_t image_slow = h5read_get_image_slow(obj);
    uint16_t image_fast = h5read_get_image_fast(obj);

    std::unordered_map<int, int> total_counts_by_image;

    for (size_t j = 0; j < 5; j++) {
        image_t *image = h5read_get_image(obj, j);
        int total = 0;
        for (int index = 0; index<image_fast*image_slow; index++){
            if (image->mask[index] != 0){
                total += image->data[index];
            }
        }
        total_counts_by_image[j] = total;
    }
    EXPECT_EQ(total_counts_by_image[0], 504245);
    EXPECT_EQ(total_counts_by_image[1], 503554);
    EXPECT_EQ(total_counts_by_image[2], 506110);
    EXPECT_EQ(total_counts_by_image[3], 505984);
    EXPECT_EQ(total_counts_by_image[4], 502084);
}


