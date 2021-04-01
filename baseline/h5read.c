#include <hdf5.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "miniapp.h"
#include "eiger2xe.h"

uint8_t *mask;
uint8_t *module_mask;
size_t mask_size;

hid_t master;
hid_t data;
hid_t dataset;

void cleanup_hdf5() {
    H5Dclose(dataset);
    H5Fclose(data);
    H5Fclose(master);
    free(mask);
    free(module_mask);
}

size_t frames, slow, fast;

size_t get_number_of_images() {
    return frames;
}

size_t get_image_slow() {
    return slow;
}

size_t get_image_fast() {
    return fast;
}

void free_image(image_t i) {
    free(i.data);
}

/* blit the relevent pixel data across from a single image into an collection
   of image modules - will allocate the latter */
void blit(image_t image, image_modules_t *modules) {
    size_t fast, slow, offset, target;

    if (image.slow == E2XE_16M_SLOW) {
        fast = 4;
        slow = 8;
    } else {
        fast = 2;
        slow = 4;
    }

    modules->mask = module_mask;
    modules->slow = E2XE_MOD_SLOW;
    modules->fast = E2XE_MOD_FAST;
    modules->modules = slow * fast;

    size_t module_pixels = E2XE_MOD_SLOW * E2XE_MOD_FAST;

    modules->data = (uint16_t *)malloc(sizeof(uint16_t) * slow * fast * module_pixels);

    for (size_t _slow = 0; _slow < slow; _slow++) {
        size_t row0 = _slow * (E2XE_MOD_SLOW + E2XE_GAP_SLOW) * image.fast;
        for (size_t _fast = 0; _fast < fast; _fast++) {
            for (size_t row = 0; row < E2XE_MOD_SLOW; row++) {
                offset =
                  (row0 + row * image.fast + _fast * (E2XE_MOD_FAST + E2XE_GAP_FAST));
                target = (_slow * fast + _fast) * module_pixels + row * E2XE_MOD_FAST;
                memcpy((void *)&modules->data[target],
                       (void *)&image.data[offset],
                       sizeof(uint16_t) * E2XE_MOD_FAST);
            }
        }
    }
}

image_modules_t get_image_modules(size_t n) {
    image_t image = get_image(n);
    image_modules_t modules;
    modules.data = NULL;
    modules.mask = NULL;
    modules.modules = -1;
    modules.fast = -1;
    modules.slow = -1;
    blit(image, &modules);
    free_image(image);
    return modules;
}

void free_image_modules(image_modules_t i) {
    free(i.data);
}

image_t get_image(size_t n) {
    if (n >= frames) {
        fprintf(stderr, "image %ld > frames (%ld)\n", n, frames);
        exit(1);
    }

    hid_t mem_space, space, datatype;

    hsize_t block[3], offset[3];

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

// void free_image_modules(image_modules_t image);

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

    // blit mask over to module mask

    size_t fast, slow, offset, target, image_slow, image_fast, module_pixels;
    module_pixels = E2XE_MOD_FAST * E2XE_MOD_SLOW;

    if (mask_size == E2XE_16M_SLOW * E2XE_16M_FAST) {
        slow = 8;
        fast = 4;
        image_slow = E2XE_16M_SLOW;
        image_fast = E2XE_16M_FAST;
    } else {
        slow = 4;
        fast = 2;
        image_slow = E2XE_4M_SLOW;
        image_fast = E2XE_4M_FAST;
    }
    module_mask = (uint8_t *)malloc(sizeof(uint8_t) * fast * slow * module_pixels);
    for (size_t _slow = 0; _slow < slow; _slow++) {
        size_t row0 = _slow * (E2XE_MOD_SLOW + E2XE_GAP_SLOW) * image_fast;
        for (size_t _fast = 0; _fast < fast; _fast++) {
            for (size_t row = 0; row < E2XE_MOD_SLOW; row++) {
                offset =
                  (row0 + row * image_fast + _fast * (E2XE_MOD_FAST + E2XE_GAP_FAST));
                target = (_slow * fast + _fast) * module_pixels + row * E2XE_MOD_FAST;
                memcpy((void *)&module_mask[target],
                       (void *)&mask[offset],
                       sizeof(uint8_t) * E2XE_MOD_FAST);
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
