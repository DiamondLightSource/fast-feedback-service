#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cinttypes>

#include "baseline.h"
#include "h5read.h"

int main(int argc, char **argv) {
    h5read_handle *obj = h5read_parse_standard_args(argc, argv);
    size_t n_images = h5read_get_number_of_images(obj);

    uint16_t image_slow = h5read_get_image_slow(obj);
    uint16_t image_fast = h5read_get_image_fast(obj);
    void *spotfinder = spotfinder_create(image_fast, image_slow);

    bool failed = false;
    for (size_t j = 0; j < n_images; j++) {
        image_t *image = h5read_get_image(obj, j);
        image_modules_t *modules = h5read_get_image_modules(obj, j);

        uint32_t strong_pixels = spotfinder_standard_dispersion(spotfinder, image);

        size_t zero = 0;
        size_t masked = 0;
        size_t total = 0;
        for (size_t i = 0; i < (image->fast * image->slow); i++) {
            ++total;
            if (image->data[i] == 0 && image->mask[i] != 0) {
                zero++;
            }
            if (image->mask[i] == 0) masked++;
        }

        size_t zero_m = 0;
        size_t masked_m = 0;
        size_t total_m = 0;
        for (size_t i = 0; i < (modules->fast * modules->slow * modules->modules);
             i++) {
            ++total_m;
            if (modules->data[i] == 0 && modules->mask[i] != 0) {
                zero_m++;
            }
            if (modules->mask[i] == 0) ++masked_m;
        }

        // Both methods should get the same number of unmasked pixels
        assert(total - masked == total_m - masked_m);

        auto col_mod = zero == zero_m ? "\033[32m" : "\033[1;31m";
        printf(
          "Image %ld had %s(%ld / %ld from modules)\033[0m valid zero pixels, %" PRIu32
          " strong pixels\n",
          j,
          col_mod,
          zero,
          zero_m,
          strong_pixels);
        if (zero != zero_m) {
            failed = true;
            printf(
              "    \033[1;31mError: Module zero count disagrees with non-module "
              "%d\033[0m\n");
        }
        h5read_free_image_modules(modules);
        h5read_free_image(image);
    }
    spotfinder_free(spotfinder);
    h5read_free(obj);

    return failed;
}
