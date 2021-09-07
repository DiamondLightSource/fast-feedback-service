#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "baseline.h"
#include "h5read.h"

int main(int argc, char **argv) {
    h5read_handle *obj = h5read_parse_standard_args(argc, argv);
    size_t n_images = h5read_get_number_of_images(obj);

    uint16_t image_slow = 0, image_fast = 0;
    void *spotfinder = NULL;
    for (size_t j = 0; j < n_images; j++) {
        image_t *image = h5read_get_image(obj, j);
        image_modules_t *modules = h5read_get_image_modules(obj, j);

        if (j == 0) {
            // Need to wait until we have an image to get its size
            image_slow = image->slow;
            image_fast = image->fast;
            spotfinder = spotfinder_create(image_fast, image_slow);
        } else {
            // For sanity sake, check this matches
            assert(image.slow == image_slow);
            assert(image.fast == image_fast);
        }

        uint32_t strong_pixels = spotfinder_standard_dispersion(spotfinder, image);

        size_t zero = 0;
        for (size_t i = 0; i < (image->fast * image->slow); i++) {
            if (image->data[i] == 0 && image->mask[i] == 1) {
                zero++;
            }
        }

        size_t zero_m = 0;
        for (size_t i = 0; i < (modules->fast * modules->slow * modules->modules);
             i++) {
            if (modules->data[i] == 0 && modules->mask[i] == 1) {
                zero_m++;
            }
        }

        printf("image %ld had %ld / %ld valid zero pixels, %" PRIu32 " strong pixels\n",
               j,
               zero,
               zero_m,
               strong_pixels);

        h5read_free_image_modules(modules);
        h5read_free_image(image);
    }
    spotfinder_free(spotfinder);
    h5read_free(obj);

    return 0;
}
