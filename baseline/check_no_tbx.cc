#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cinttypes>

#include "baseline.h"
#include "h5read.h"
#include "no_tbx.h"

int main(int argc, char **argv) {
    h5read_handle *obj = h5read_parse_standard_args(argc, argv);
    size_t n_images = h5read_get_number_of_images(obj);

    uint16_t image_slow = 0, image_fast = 0;
    void *spotfinder = nullptr;
    void *no_tbx_spotfinder = nullptr;
    bool *strong_spotfinder = nullptr;
    bool *strong_no_tbx_spotfinder = nullptr;

    for (size_t j = 0; j < n_images; j++) {
        image_t *image = h5read_get_image(obj, j);
        image_modules_t *modules = h5read_get_image_modules(obj, j);

        if (j == 0) {
            // Need to wait until we have an image to get its size
            image_slow = image->slow;
            image_fast = image->fast;
            spotfinder = spotfinder_create(image_fast, image_slow);
            no_tbx_spotfinder = no_tbx_spotfinder_create(image_fast, image_slow);
        }

        uint32_t strong_pixels =
          spotfinder_standard_dispersion(spotfinder, image, &strong_spotfinder);
        uint32_t strong_pixels_no_tbx = no_tbx_spotfinder_standard_dispersion(
          spotfinder, image, &strong_no_tbx_spotfinder);

        size_t zero = 0;
        for (size_t i = 0; i < (image->fast * image->slow); i++) {
            if (image->data[i] == 0 && image->mask[i] == 1) {
                zero++;
            }
            if (strong_spotfinder[i] != strong_no_tbx_spotfinder[i]) {
                printf("Error: Spotfinders disagree at %d\n", int(i));
                exit(1);
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
