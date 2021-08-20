#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h5read.h"

const char *USAGE = "Usage: %s [-h|--help] [FILE.nxs]\n";

int main(int argc, char **argv) {
    // Handle simple case of -h or --help
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr, USAGE, argv[0]);
            return 0;
        }
    }
    if (argc == 1) {
        fprintf(stderr, USAGE, argv[0]);
        return 1;
    }

    h5read_handle *obj = h5read_open(argv[1]);
    if (obj == NULL) {
        fprintf(stderr, "<shrug> bad thing </shrug>\n");
        exit(1);
    }

    size_t n_images = h5read_get_number_of_images(obj);

    for (size_t j = 0; j < n_images; j++) {
        image_t *image = h5read_get_image(obj, j);
        image_modules_t *modules = h5read_get_image_modules(obj, j);

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

        printf("image %ld had %ld / %ld valid zero pixels\n", j, zero, zero_m);

        h5read_free_image_modules(modules);
        h5read_free_image(image);
    }

    h5read_free(obj);

    return 0;
}
