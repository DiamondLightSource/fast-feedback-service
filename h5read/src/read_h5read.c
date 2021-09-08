#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h5read.h"

int main(int argc, char **argv) {
    h5read_handle *obj = h5read_parse_standard_args(argc, argv);

    if (obj == NULL) {
        fprintf(stderr, "Error: Failed to open %s\n", argv[1]);
        exit(1);
    }

    size_t n_images = h5read_get_number_of_images(obj);

    printf("               %8s / %s\n", "Image", "Module");
    for (size_t j = 0; j < n_images; j++) {
        image_t *image = h5read_get_image(obj, j);
        image_modules_t *modules = h5read_get_image_modules(obj, j);

        size_t zero = 0, zero_invalid = 0;
        for (size_t i = 0; i < (image->fast * image->slow); i++) {
            if (image->data[i] == 0) {
                if (image->mask[i] == 1) {
                    zero++;
                } else {
                    zero_invalid++;
                }
            }
        }

        size_t zero_m = 0;
        for (size_t i = 0; i < (modules->fast * modules->slow * modules->modules);
             i++) {
            if (modules->data[i] == 0 && modules->mask[i] == 1) {
                zero_m++;
            }
        }
        char *colour = "\033[31m";
        if (zero == zero_m) {
            colour = "\033[32m";
        }
        printf(
          "%simage %4ld had %8ld / %8ld valid zero pixels (%zd invalid zero "
          "pixel)\033[0m\n",
          colour,
          j,
          zero,
          zero_m,
          zero_invalid);

        h5read_free_image_modules(modules);
        h5read_free_image(image);
    }

    h5read_free(obj);

    return 0;
}
