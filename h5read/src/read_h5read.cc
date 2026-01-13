#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "h5read.h"

int main(int argc, char **argv) {
    auto reader = H5Read(argc, argv);

    size_t n_images = reader.get_number_of_images();
    size_t elem_size = reader.get_element_size();
    h5read_dtype dtype = reader.get_data_dtype();
    size_t num_pixels = reader.get_image_slow() * reader.get_image_fast();

    printf("Data type: %d, element size: %zu bytes\n", dtype, elem_size);

    // A buffer we own, to check reading image data into a preallocated buffer
    auto buffer = std::make_unique<uint8_t[]>(num_pixels * elem_size);

    auto [px_min, px_max] = reader.get_trusted_range();
    printf(
      "Trusted pixel inclusive range: %" PRId64 " → %" PRId64 "\n", px_min, px_max);

    printf("               %8s / %s\n", "Image", "Module");
    for (size_t j = 0; j < n_images; j++) {
        auto image = reader.get_image(j);
        auto modules = reader.get_image_modules(j);
        reader.get_image_into(j, buffer.get());

        size_t zero = 0, zero_invalid = 0;

        // Verify buffer matches image data
        assert(std::memcmp(buffer.get(), image.data(), num_pixels * elem_size) == 0);

        // Count zeros based on dtype
        if (dtype == H5READ_DTYPE_UINT16) {
            uint16_t *data = static_cast<uint16_t *>(image.data());
            for (size_t i = 0; i < image.slow * image.fast; i++) {
                if (data[i] == 0) {
                    if (image.mask[i] == 1) {
                        zero++;
                    } else {
                        zero_invalid++;
                    }
                }
            }
        } else if (dtype == H5READ_DTYPE_UINT32) {
            uint32_t *data = static_cast<uint32_t *>(image.data());
            for (size_t i = 0; i < image.slow * image.fast; i++) {
                if (data[i] == 0) {
                    if (image.mask[i] == 1) {
                        zero++;
                    } else {
                        zero_invalid++;
                    }
                }
            }
        }

        size_t zero_m = 0;
        if (dtype == H5READ_DTYPE_UINT16) {
            uint16_t *mdata = static_cast<uint16_t *>(modules.data());
            for (size_t i = 0; i < modules.slow * modules.fast * modules.n_modules;
                 i++) {
                if (mdata[i] == 0 && modules.mask[i] == 1) {
                    zero_m++;
                }
            }
        } else if (dtype == H5READ_DTYPE_UINT32) {
            uint32_t *mdata = static_cast<uint32_t *>(modules.data());
            for (size_t i = 0; i < modules.slow * modules.fast * modules.n_modules;
                 i++) {
                if (mdata[i] == 0 && modules.mask[i] == 1) {
                    zero_m++;
                }
            }
        }

        if (zero == zero_m) {
            std::cout << "\033[32m";
        } else {
            std::cout << "\033[31m";
        }
        printf(
          "Image %4zu had %8zu / %8zu valid zero pixels (%zu invalid zero "
          "pixel)\033[0m\n",
          j,
          zero,
          zero_m,
          zero_invalid);
    }
}
