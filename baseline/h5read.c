#include <hdf5.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "miniapp.h"

uint8_t *mask;
size_t mask_size;

hid_t master;
hid_t data;
hid_t dataset;

size_t frames, slow, fast;

size_t get_number_of_images() {
    return frames;
}

void free_image(image_t i) {
    free(i.data);
}

image_t get_image(size_t n) {
    if (n >= frames) {
        fprintf(stderr, "image %ld > frames (%ld)\n", n, frames);
        exit(1);
    }

    hid_t mem_space, space, datatype;

    hsize_t dims[3], block[3], offset[3];

    uint16_t *buffer = (uint16_t *)malloc(sizeof(uint16_t) * slow * fast);

    block[0] = 1;
    block[1] = slow;
    block[2] = fast;

    offset[0] = n;
    offset[1] = 0;
    offset[2] = 0;

    space = H5Dget_space(dataset);
    datatype = H5Dget_type(dataset);

    // select data to read #todo add status checks
    H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, NULL, block, NULL);
    mem_space = H5Screate_simple(3, block, NULL);

    H5Dread(dataset, datatype, mem_space, space, H5P_DEFAULT, buffer);

    H5Sclose(space);
    H5Sclose(mem_space);

    image_t result;
    result.slow = slow;
    result.fast = fast;
    result.mask = mask;
    result.data = buffer;

    return result;
}

void read_mask() {
    // uses master pointer above: beware if this is bad

    char mask_path[] = "/entry/instrument/detector/pixel_mask";
    hid_t mask_dataset, mask_info, datatype;

    size_t mask_dsize;
    uint32_t *raw_mask;
    uint64_t *raw_mask_64;  // why?

    mask_dataset = H5Dopen(master, mask_path, H5P_DEFAULT);

    if (mask_dataset < 0) {
        fprintf(stderr, "error reading mask from %s\n", mask_path);
        exit(1);
    }

    datatype = H5Dget_type(mask_dataset);
    mask_info = H5Dget_space(mask_dataset);

    mask_dsize = H5Tget_size(datatype);
    if (mask_dsize == 4) {
        printf("mask dtype uint32");
    } else if (mask_dsize == 8) {
        printf("mask dtype uint64");
    } else {
        fprintf(stderr, "mask data size != 4,8 (%ld)\n", H5Tget_size(datatype));
        exit(1);
    }

    mask_size = H5Sget_simple_extent_npoints(mask_info);

    printf("mask has %ld elements\n", mask_size);

    void *buffer = NULL;

    if (mask_dsize == 4) {
        raw_mask = (uint32_t *)malloc(sizeof(uint32_t) * mask_size);
        buffer = (void *)raw_mask;
        raw_mask_64 = NULL;
    } else {
        raw_mask_64 = (uint64_t *)malloc(sizeof(uint64_t) * mask_size);
        buffer = (void *)raw_mask_64;
        raw_mask = NULL;
    }

    if (H5Dread(mask_dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) < 0) {
        fprintf(stderr, "error reading mask\n");
        exit(1);
    }

    // count 0's

    size_t zero = 0;

    mask = (uint8_t *)malloc(sizeof(uint8_t) * mask_size);

    if (mask_dsize == 4) {
        for (size_t j = 0; j < mask_size; j++) {
            if (raw_mask[j] == 0) {
                zero++;
                mask[j] = 1;
            } else {
                mask[j] = 0;
            }
        }
    } else {
        for (size_t j = 0; j < mask_size; j++) {
            if (raw_mask_64[j] == 0) {
                zero++;
                mask[j] = 1;
            } else {
                mask[j] = 0;
            }
        }
    }

    printf("%ld of the pixels are valid\n", zero);

    // cleanup

    if (raw_mask) free(raw_mask);
    if (raw_mask_64) free(raw_mask_64);
    H5Dclose(mask_dataset);
}

void setup_data() {
    // uses master pointer above: beware if this is bad

    char data_path[] = "/data";

    hid_t datatype, space;

    hsize_t dims[3];

    dataset = H5Dopen(data, data_path, H5P_DEFAULT);

    if (dataset < 0) {
        fprintf(stderr, "error reading data from %s\n", data_path);
        exit(1);
    }

    datatype = H5Dget_type(dataset);

    if (H5Tget_size(datatype) != 2) {
        fprintf(stderr, "native data size != 2 (%ld)\n", H5Tget_size(datatype));
        exit(1);
    }

    space = H5Dget_space(dataset);

    if (H5Sget_simple_extent_ndims(space) != 3) {
        fprintf(stderr, "raw data not three dimensional\n");
        exit(1);
    }

    H5Sget_simple_extent_dims(space, dims, NULL);

    frames = dims[0];
    slow = dims[1];
    fast = dims[2];

    printf("total data size: %ldx%ldx%ld\n", frames, slow, fast);
}

int setup_hdf5_files(char *master_filename, char *data_filename) {
    /* I'll do my own debug printing: disable HDF5 library output */
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    master = H5Fopen(master_filename, H5F_ACC_RDONLY | H5F_ACC_SWMR_READ, H5P_DEFAULT);

    if (master < 0) {
        fprintf(stderr, "error reading %s\n", master_filename);
        return 1;
    }

    data = H5Fopen(data_filename, H5F_ACC_RDONLY | H5F_ACC_SWMR_READ, H5P_DEFAULT);

    if (data < 0) {
        fprintf(stderr, "error reading %s\n", data_filename);
        return 1;
    }

    read_mask();

    setup_data();

    // do stuff

    // cleanup

    return 0;
}
