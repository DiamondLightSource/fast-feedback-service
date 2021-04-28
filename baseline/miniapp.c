#include "miniapp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *USAGE = "Usage: %s [-h|--help] [FILE.nxs]\n";

int main(int argc, char **argv) {
    // Handle simple case of -h or --help
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") || strcmp(argv[i], "--help")) {
            fprintf(stderr, USAGE, argv[0]);
            return 0;
        }
    }
    if (argc == 1) {
        fprintf(stderr, USAGE, argv[0]);
        return 1;
    }

    if (setup_hdf5_files(argv[1]) < 0) {
        fprintf(stderr, "<shrug> bad thing </shrug>\n");
        exit(1);
    }

    size_t n_images = get_number_of_images();

    for (size_t j = 0; j < n_images; j++) {
        image_t image = get_image(j);
        image_modules_t modules = get_image_modules(j);

        size_t zero = 0;
        for (size_t i = 0; i < (image.fast * image.slow); i++) {
            if (image.data[i] == 0 && image.mask[i] == 1) {
                zero++;
            }
        }

        size_t zero_m = 0;
        for (size_t i = 0; i < (modules.fast * modules.slow * modules.modules); i++) {
            if (modules.data[i] == 0 && modules.mask[i] == 1) {
                zero_m++;
            }
        }

        printf("image %ld had %ld / %ld valid zero pixels\n", j, zero, zero_m);

        free_image_modules(modules);
        free_image(image);
    }

    cleanup_hdf5();

    return 0;
}
