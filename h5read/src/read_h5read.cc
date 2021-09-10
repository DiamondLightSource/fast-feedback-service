#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>

#include "h5read.h"

int main(int argc, char **argv) {
    auto reader = H5Read(argc, argv);

    size_t n_images = reader.get_number_of_images();

    // A buffer we own, to check reading image data into a preallocated buffer
    auto buffer = std::make_unique<H5Read::image_type[]>(reader.get_image_slow()
                                                         * reader.get_image_fast());

    printf("               %8s / %s\n", "Image", "Module");
    for (size_t j = 0; j < n_images; j++) {
        auto image = reader.get_image(j);
        auto modules = reader.get_image_modules(j);
        reader.get_image_into(j, buffer.get());

        size_t zero = 0, zero_invalid = 0;
        for (size_t i = 0; i < image.data.size(); i++) {
            assert(image.data[i] == buffer[i]);
            if (image.data[i] == 0) {
                if (image.mask[i] == 1) {
                    zero++;
                } else {
                    zero_invalid++;
                }
            }
        }

        size_t zero_m = 0;
        for (size_t i = 0; i < modules.data.size(); i++) {
            if (modules.data[i] == 0 && modules.mask[i] == 1) {
                zero_m++;
            }
        }
        if (zero == zero_m) {
            std::cout << "\033[32m";
        } else {
            std::cout << "\033[31m";
        }
        printf(
          "Image %4ld had %8ld / %8ld valid zero pixels (%zd invalid zero "
          "pixel)\033[0m\n",
          j,
          zero,
          zero_m,
          zero_invalid);
    }
}
