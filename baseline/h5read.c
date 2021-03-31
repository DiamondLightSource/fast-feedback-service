#include <hdf5.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

uint8_t *mask;
size_t mask_size;

hid_t master;
hid_t dataset;

typedef struct image {
    uint16_t *data;
    uint8_t *mask;
    size_t n_slow;
    size_t n_fast;
} image;

void read_mask() {
    // uses master pointer above: beware if this is bad

    char mask_path[] = "/entry/instrument/detector/pixel_mask";
    hid_t mask_dataset, mask_info, datatype;

    size_t mask_dsize;
    uint32_t *raw_mask;
    uint64_t *raw_mask_64; // why?

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
      buffer = (void *) raw_mask;
      raw_mask_64 = NULL;
    } else {
      raw_mask_64 = (uint64_t *)malloc(sizeof(uint64_t) * mask_size);
      buffer = (void *) raw_mask_64;
      raw_mask = NULL;
    }
    
    if (H5Dread(mask_dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer)
        < 0) {
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

    char data_path[] = "/entry/data/data";

    hid_t datatype;

    dataset = H5Dopen(master, data_path, H5P_DEFAULT);

    if (dataset < 0) {
        fprintf(stderr, "error reading data from %s\n", data_path);
        exit(1);
    }

    datatype = H5Dget_type(dataset);

    if (H5Tget_size(datatype) != 2) {
        fprintf(stderr, "native data size != 2 (%ld)\n", H5Tget_size(datatype));
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc == 1) {
        fprintf(stderr, "%s /path/to/foobar.nxs\n", argv[0]);
        return 1;
    }

    /* I'll do my own debug printing: disable HDF5 library output */
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    master = H5Fopen(argv[1], H5F_ACC_RDONLY | H5F_ACC_SWMR_READ, H5P_DEFAULT);

    if (master < 0) {
        fprintf(stderr, "error reading %s\n", argv[1]);
        return 1;
    }

    read_mask();

    setup_data();

    // do stuff

    // cleanup

    return 0;
}
