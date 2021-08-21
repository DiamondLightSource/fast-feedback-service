#include <hdf5.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h5read.h"

const char *USAGE = "Usage: %s [-h|--help] [-v] [FILE.nxs]\n";

int main(int argc, char **argv) {
    bool verbose = false;
    // Handle simple case of -h or --help
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr, USAGE, argv[0]);
            return 0;
        }
        if (!strcmp(argv[i], "-v")) {
            verbose = true;
            // Shift the rest over this one so that we only have positionals
            for (int j = i; j < argc; j++) {
                argv[i] = argv[j];
            }
            argc -= 1;
        }
    }
    if (!verbose) {
        // Turn off verbose hdf5 errors
        H5Eset_auto(H5E_DEFAULT, NULL, NULL);
    }
    if (argc == 1) {
        fprintf(stderr, USAGE, argv[0]);
        return 1;
    }
    h5read_handle *obj = h5read_open(argv[1]);
    if (obj == NULL) {
        fprintf(stderr, "Error: Failed to open %s\n", argv[1]);
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
