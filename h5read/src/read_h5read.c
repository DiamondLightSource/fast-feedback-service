#include <assert.h>
#include <inttypes.h>
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
    size_t elem_size = h5read_get_element_size(obj);
    h5read_dtype dtype = h5read_get_data_dtype(obj);

    printf("Data type: %d, element size: %zu bytes\n", dtype, elem_size);

    // A buffer we own, to check reading image data into a preallocated buffer
    size_t num_pixels = h5read_get_image_fast(obj) * h5read_get_image_slow(obj);
    void *buffer = malloc(num_pixels * elem_size);

    int64_t max, min;
    h5read_get_trusted_range(obj, &min, &max);
    printf("Trusted pixel inclusive range: %" PRId64 " → %" PRId64 "\n", min, max);
    printf("               %8s / %s\n", "Image", "Module");
    for (size_t j = 0; j < n_images; j++) {
        image_t *image = h5read_get_image(obj, j);
        image_modules_t *modules = h5read_get_image_modules(obj, j);
        h5read_get_image_into(obj, j, buffer);
        size_t zero = 0, zero_invalid = 0;

        // Compare data based on detected dtype
        if (dtype == H5READ_DTYPE_UINT16) {
            uint16_t *buf = (uint16_t *)buffer;
            uint16_t *data = (uint16_t *)image->data;
            for (size_t i = 0; i < (image->fast * image->slow); i++) {
                // Check that our owned buffer is correct
                assert(buf[i] == data[i]);
                if (data[i] == 0) {
                    if (image->mask[i] == 1) {
                        zero++;
                    } else {
                        zero_invalid++;
                    }
                }
            }
        } else if (dtype == H5READ_DTYPE_UINT32) {
            uint32_t *buf = (uint32_t *)buffer;
            uint32_t *data = (uint32_t *)image->data;
            for (size_t i = 0; i < (image->fast * image->slow); i++) {
                assert(buf[i] == data[i]);
                if (data[i] == 0) {
                    if (image->mask[i] == 1) {
                        zero++;
                    } else {
                        zero_invalid++;
                    }
                }
            }
        } else {
            // For other types, just verify buffer copy worked via memcmp
            assert(memcmp(buffer, image->data, num_pixels * elem_size) == 0);
            printf("image %4zu: data type %d not fully tested\n", j, dtype);
        }

        size_t zero_m = 0;
        if (dtype == H5READ_DTYPE_UINT16) {
            uint16_t *mdata = (uint16_t *)modules->data;
            for (size_t i = 0; i < (modules->fast * modules->slow * modules->modules);
                 i++) {
                if (mdata[i] == 0 && modules->mask[i] == 1) {
                    zero_m++;
                }
            }
        } else if (dtype == H5READ_DTYPE_UINT32) {
            uint32_t *mdata = (uint32_t *)modules->data;
            for (size_t i = 0; i < (modules->fast * modules->slow * modules->modules);
                 i++) {
                if (mdata[i] == 0 && modules->mask[i] == 1) {
                    zero_m++;
                }
            }
        }

        char *colour = "\033[31m";
        if (zero == zero_m) {
            colour = "\033[32m";
        }
        printf(
          "%simage %4zu had %8zu / %8zu valid zero pixels (%zu invalid zero "
          "pixel)\033[0m\n",
          colour,
          j,
          zero,
          zero_m,
          zero_invalid);

        h5read_free_image_modules(modules);
        h5read_free_image(image);
    }

    free(buffer);
    h5read_free(obj);

    return 0;
}
