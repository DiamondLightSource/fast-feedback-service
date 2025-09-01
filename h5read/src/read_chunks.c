#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h5read.h"

int main(int argc, char** argv) {
    h5read_handle* obj = h5read_parse_standard_args(argc, argv);

    if (obj == NULL) {
        fprintf(stderr, "Error: Failed to open %s\n", argv[1]);
        exit(1);
    }

    size_t n_images = h5read_get_number_of_images(obj);

    // A buffer we own, to check reading image data into a preallocated buffer
    image_t_type* buffer = malloc(h5read_get_image_fast(obj)
                                  * h5read_get_image_slow(obj) * sizeof(image_t_type));
    image_t_type* buffer_manual = malloc(
      h5read_get_image_fast(obj) * h5read_get_image_slow(obj) * sizeof(image_t_type));

    size_t max_compressed_bytes =
      h5read_get_image_fast(obj) * h5read_get_image_slow(obj) * sizeof(image_t_type);
    uint8_t* chunk_data = malloc(max_compressed_bytes);

    // printf("               %8s / %s\n", "Image", "Module");
    for (size_t j = 0; j < n_images; j++) {
        // image_t *image = h5read_get_image(obj, j);
        // h5read_get_image_into(obj, j, buffer);
        size_t data_size = 0;
        h5read_get_raw_chunk(obj, j, &data_size, chunk_data, max_compressed_bytes);
        printf("Read Image %d chunk in %zu KBytes\n", j, data_size / 1024);

        bool success = true;
        char* colour = "\033[31m";
        if (success) {
            colour = "\033[32m";
        }
    }

    free(chunk_data);
    free(buffer);
    h5read_free(obj);

    return 0;
}
