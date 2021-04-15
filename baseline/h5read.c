#include <hdf5.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "eiger2xe.h"
#include "miniapp.h"

uint8_t *mask;
uint8_t *module_mask;
size_t mask_size;

// VDS stuff

#define MAXFILENAME 256
#define MAXDATAFILES 100
#define MAXDIM 3

typedef struct h5_data_file {
    char filename[MAXFILENAME];
    char dsetname[MAXFILENAME];
    hid_t file;
    hid_t dataset;
    size_t frames;
    size_t offset;
} h5_data_file;

// allocate space for 100 virtual files (which would mean 100,000 frames)

h5_data_file data_files[MAXDATAFILES];
int data_file_count;
int data_file_current;

hid_t master;

void cleanup_hdf5() {
    for (int i = 0; i < data_file_count; i++) {
        H5Dclose(data_files[i].dataset);
        H5Fclose(data_files[i].file);
    }
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
    int data_file;

    h5_data_file *current;

    if (n >= frames) {
        fprintf(stderr, "image %ld > frames (%ld)\n", n, frames);
        exit(1);
    }

    /* first find the right data file - having to do this lookup is annoying
       but probably cheap */

    for (data_file = 0; data_file < data_file_count; data_file++) {
        if ((n - data_files[data_file].offset) < data_files[data_file].frames) {
            break;
        }
    }

    if (data_file == data_file_count) {
        fprintf(stderr, "could not find data file for frame %ld\n", n);
        exit(1);
    }

    current = &(data_files[data_file]);

    hid_t mem_space, space, datatype;

    hsize_t block[3], offset[3];

    uint16_t *buffer = (uint16_t *)malloc(sizeof(uint16_t) * slow * fast);

    block[0] = 1;
    block[1] = slow;
    block[2] = fast;

    offset[0] = n - current->offset;
    offset[1] = 0;
    offset[2] = 0;

    space = H5Dget_space(current->dataset);
    datatype = H5Dget_type(current->dataset);

    // select data to read #todo add status checks
    H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, NULL, block, NULL);
    mem_space = H5Screate_simple(3, block, NULL);

    if (H5Dread(current->dataset, datatype, mem_space, space, H5P_DEFAULT, buffer)
        < 0) {
        H5Eprint(H5E_DEFAULT, NULL);
        exit(1);
    }

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

int vds_info(char *root, hid_t master, hid_t dataset, h5_data_file *vds) {
    hid_t plist, vds_source;
    size_t vds_count;
    herr_t status;

    plist = H5Dget_create_plist(dataset);

    status = H5Pget_virtual_count(plist, &vds_count);

    for (int j = 0; j < vds_count; j++) {
        hsize_t start[MAXDIM], stride[MAXDIM], count[MAXDIM], block[MAXDIM];
        size_t dims;

        vds_source = H5Pget_virtual_vspace(plist, j);
        dims = H5Sget_simple_extent_ndims(vds_source);

        if (dims != 3) {
            H5Sclose(vds_source);
            fprintf(stderr, "incorrect data dimensionality: %d\n", (int)dims);
            return -1;
        }

        H5Sget_regular_hyperslab(vds_source, start, stride, count, block);
        H5Sclose(vds_source);

        H5Pget_virtual_filename(plist, j, vds[j].filename, MAXFILENAME);
        H5Pget_virtual_dsetname(plist, j, vds[j].dsetname, MAXFILENAME);

        for (int k = 1; k < dims; k++) {
            if (start[k] != 0) {
                fprintf(stderr, "incorrect chunk start: %d\n", (int)start[k]);
                return -1;
            }
        }

        vds[j].frames = block[0];
        vds[j].offset = start[0];

        if ((strlen(vds[j].filename) == 1) && (vds[j].filename[0] == '.')) {
            H5L_info_t info;
            status = H5Lget_info(master, vds[j].dsetname, &info, H5P_DEFAULT);

            if (status) {
                fprintf(stderr, "error from H5Lget_info on %s\n", vds[j].dsetname);
                return -1;
            }

            /* if the data file points to an external source, dereference */

            if (info.type == H5L_TYPE_EXTERNAL) {
                char buffer[MAXFILENAME], scr[MAXFILENAME];
                unsigned flags;
                const char *nameptr, *dsetptr;

                H5Lget_val(master, vds[j].dsetname, buffer, MAXFILENAME, H5P_DEFAULT);
                H5Lunpack_elink_val(
                  buffer, info.u.val_size, &flags, &nameptr, &dsetptr);

                /* assumptions herein:
                    - external link references are local paths
                    - only need to worry about UNIX paths e.g. pathsep is /
                    - ASCII so chars are ... chars
                   so manually assemble...
                 */

                strcpy(scr, root);
                scr[strlen(root)] = '/';
                strcpy(scr + strlen(root) + 1, nameptr);

                strcpy(vds[j].filename, scr);
                strcpy(vds[j].dsetname, dsetptr);
            }
        } else {
            char scr[MAXFILENAME];
            sprintf(scr, "%s/%s", root, vds[j].filename);
            strcpy(vds[j].filename, scr);
        }

        // do I want to open these here? Or when they are needed...
        vds[j].file = 0;
        vds[j].dataset = 0;
    }

    status = H5Pclose(plist);

    return vds_count;
}

int unpack_vds(char *filename, h5_data_file *data_files) {
    hid_t dataset, file;
    char *root, cwd[MAXFILENAME];
    int retval;

    // TODO if we want this to become SWMR aware in the future will need to
    // allow for that here
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    if (file < 0) {
        fprintf(stderr, "error reading %s\n", filename);
        return -1;
    }

    dataset = H5Dopen(file, "/entry/data/data", H5P_DEFAULT);

    if (dataset < 0) {
        H5Fclose(file);
        fprintf(stderr, "error reading %s\n", "/entry/data/data");
        return -1;
    }

    /* always set the absolute path to file information */
    root = dirname(filename);
    if ((strlen(root) == 1) && (root[0] == '.')) {
        root = getcwd(cwd, MAXFILENAME);
    }

    retval = vds_info(root, file, dataset, data_files);

    H5Dclose(dataset);
    H5Fclose(file);

    return retval;
}

void setup_data() {
    hid_t datatype, space, dataset;

    hsize_t dims[3];

    dataset = data_files[0].dataset;

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

    slow = dims[1];
    fast = dims[2];

    printf("total data size: %ldx%ldx%ld\n", frames, slow, fast);
    H5Sclose(space);
}

int setup_hdf5_files(char *master_filename) {
    /* I'll do my own debug printing: disable HDF5 library output */
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    master = H5Fopen(master_filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    if (master < 0) {
        fprintf(stderr, "error reading %s\n", master_filename);
        return 1;
    }

    data_file_count = unpack_vds(master_filename, data_files);

    if (data_file_count < 0) {
        fprintf(stderr, "error reading %s\n", master_filename);
        return 1;
    }

    // open up the actual data files, count all the frames
    frames = 0;
    for (int j = 0; j < data_file_count; j++) {
        data_files[j].file =
          H5Fopen(data_files[j].filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (data_files[j].file < 0) {
            fprintf(stderr, "error reading %s\n", data_files[j].filename);
            return 1;
        }
        data_files[j].dataset =
          H5Dopen(data_files[j].file, data_files[j].dsetname, H5P_DEFAULT);
        if (data_files[j].dataset < 0) {
            fprintf(stderr, "error reading %s\n", data_files[j].filename);
            return 1;
        }
        frames += data_files[j].frames;
    }

    read_mask();

    setup_data();

    return 0;
}
