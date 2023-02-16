#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cinttypes>

#include "baseline.h"
#include "h5read.h"
#include "standalone.h"

int main(int argc, char **argv) {
    auto reader = H5Read(argc, argv);
    size_t n_images = reader.get_number_of_images();

    auto [image_slow, image_fast] = reader.image_shape();

    bool *strong_spotfinder = nullptr;
    auto *spotfinder = spotfinder_create(image_fast, image_slow);
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
        // uint32_t strong_pixels_no_tbx = no_tbx_spotfinder_standard_dispersion(
        //   no_tbx_spotfinder, image, &strong_no_tbx_spotfinder);
        // standalone_spotfinder.standard_dispersion(image, )
        image_double.assign(image.data.begin(), image.data.end());
        auto standalone_strong_pixels = standalone_spotfinder.standard_dispersion(
          image_double, {reinterpret_cast<bool *>(mask.data()), mask.size()});

        size_t zero = 0;
        size_t n_strong = 0;
        size_t n_strong_notbx = 0;
        long first_incorrect_index = -1;
        for (size_t i = 0; i < (image_fast * image_slow); i++) {
            if (image.data[i] == 0 && image.mask[i] == 1) {
                zero++;
            }
            if (strong_spotfinder[i]) ++n_strong;
            if (standalone_strong_pixels[i]) ++n_strong_notbx;
            if (strong_spotfinder[i] != standalone_strong_pixels[i]
                && first_incorrect_index == -1) {
                first_incorrect_index = i;
            }
        }

        size_t zero_m = 0;
        for (size_t i = 0; i < (modules.fast * modules.slow * modules.n_modules); i++) {
            if (modules.data[i] == 0 && modules.mask[i] == 1) {
                zero_m++;
            }
        }

        printf("\nImage %ld had %ld / %ld valid zero pixels, %" PRIu32
               " strong pixels\n",
               j,
               zero,
               zero_m,
               strong_pixels);
        auto col = n_strong == n_strong_notbx ? "\033[32m" : "\033[1;31m";
        printf("    %sDIALS %5d %s %-5d standalone\033[0m\n",
               col,
               (int)n_strong,
               n_strong == n_strong_notbx ? "==" : "!=",
               (int)n_strong_notbx);
        if (first_incorrect_index != -1) {
            printf("    \033[1;31mError: Spotfinders disagree at %d\033[0m\n",
                   int(first_incorrect_index));
            failed = true;
        }
    }
    spotfinder_free(spotfinder);

    return failed;
}
