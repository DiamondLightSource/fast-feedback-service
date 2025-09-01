#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cinttypes>

#include "baseline.h"
#include "common.hpp"
#include "h5read.h"
#include "standalone.h"

int main(int argc, char** argv) {
    auto reader = H5Read(argc, argv);
    size_t n_images = reader.get_number_of_images();

    auto [image_slow, image_fast] = reader.image_shape();

    bool* strong_spotfinder = nullptr;
    auto* spotfinder = spotfinder_create(image_fast, image_slow);
    auto standalone_spotfinder = StandaloneSpotfinder(image_fast, image_slow);

    auto mask = reader.get_mask().value_or(span<uint8_t>{});

    auto image_double = std::vector<double>(image_fast * image_slow);

    bool failed = false;
    for (size_t j = 0; j < n_images; j++) {
        auto image = reader.get_image(j);
        auto modules = reader.get_image_modules(j);

        // Construct an image_t for the standard_dispersion call
        image_t c_image{.data = image.data.data(),
                        .mask = mask.data(),
                        .slow = image_slow,
                        .fast = image_fast};
        uint32_t strong_pixels =
          spotfinder_standard_dispersion(spotfinder, &c_image, &strong_spotfinder);
        image_double.assign(image.data.begin(), image.data.end());
        auto standalone_strong_pixels = standalone_spotfinder.standard_dispersion(
          image_double, {reinterpret_cast<bool*>(mask.data()), mask.size()});

        size_t zero = 0;
        size_t n_strong = count_nonzero(strong_spotfinder, image_fast, image_slow);
        size_t n_strong_notbx =
          count_nonzero(standalone_strong_pixels, image_fast, image_slow);
        size_t first_incorrect_index = 0;
        // x, y positions of disagreement
        size_t mismatch_x = 0, mismatch_y = 0;

        bool result = compare_results(strong_spotfinder,
                                      image_fast,
                                      standalone_strong_pixels.data(),
                                      image_fast,
                                      image_fast,
                                      image_slow,
                                      &mismatch_x,
                                      &mismatch_y);

        // Count zeros in the image data
        for (size_t i = 0; i < (image_fast * image_slow); i++) {
            if (image.data[i] == 0 && image.mask[i] == 1) {
                zero++;
            }
        }
        // Count zeros from using masks
        size_t zero_m = 0;
        for (size_t i = 0; i < (modules.fast * modules.slow * modules.n_modules); i++) {
            if (modules.data[i] == 0 && modules.mask[i] == 1) {
                zero_m++;
            }
        }

        auto col_mod = zero == zero_m ? "\033[32m" : "\033[1;31m";
        printf(
          "\nImage %ld had %s(%ld / %ld from modules)\033[0m valid zero pixels, "
          "%" PRIu32 " strong pixels\n",
          j,
          col_mod,
          zero,
          zero_m,
          strong_pixels);
        auto col = n_strong == n_strong_notbx ? "\033[32m" : "\033[1;31m";
        printf("    %sDIALS %5d %s %-5d standalone\033[0m\n",
               col,
               (int)n_strong,
               n_strong == n_strong_notbx ? "==" : "!=",
               (int)n_strong_notbx);
        if (zero != zero_m) {
            printf(
              "    \033[1;31mError: Module zero count disagrees with non-module "
              "%d\033[0m\n");
            failed = true;
        }
        if (!result) {
            printf(
              "    \033[1;31mError: Spotfinders disagree at x, y = (%d, %d)\033[0m\n",
              int(mismatch_x),
              int(mismatch_y));
            failed = true;
        }
        if (!result) {
            // If the sources don't match, draw tables of disagreement
            size_t x = std::max(0, (int)mismatch_x - 6);
            size_t y = std::max(0, (int)mismatch_y - 6);
            fmt::print("Image Data:\n");
            draw_image_data(
              image.data, x, y, (size_t)12, (size_t)12, image_fast, image_slow);
            fmt::print("Mask:\n");
            draw_image_data(
              image.mask, x, y, (size_t)12, (size_t)12, image_fast, image_slow);

            fmt::print("C API DIALS Spotfinder:\n");
            draw_image_data(strong_spotfinder, x, y, 12, 12, image_fast, image_slow);
            fmt::print("Standalone Spotfinder:\n");
            draw_image_data(
              standalone_strong_pixels, x, y, 12, 12, image_fast, image_slow);
        }
    }
    spotfinder_free(spotfinder);

    return failed;
}
