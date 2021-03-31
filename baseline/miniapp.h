#ifndef __MINIAPP_H
#define __MINIAPP_H
#include <unistd.h>

typedef struct image_t {
    uint16_t *data;
    uint8_t *mask;
    size_t slow;
    size_t fast;
} image_t;

// file reading API specification

int setup_hdf5_files(char *master_filename, char *data_filename);
void cleanup_hdf5();

image_t get_image(size_t number);

void free_image(image_t image);

size_t get_number_of_images();

#endif
