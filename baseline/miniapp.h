#ifndef __MINIAPP_H
#define __MINIAPP_H

#include <stdlib.h>

/* Basic API specification:
 *
 * Call
 *
 * setup_hdf5_files(master_filename, data_filename)
 *
 * to initialise all of the HDF5 data structures, read and transform the mask
 * etc. Then
 *
 * get_number_of_images()
 *
 * will return the number of images in the given data file. You can then read
 * these images to return an image_t with
 *
 * get_image(number)
 *
 * which you must call free_image() on when you are done. At the end it would
 * be polite to call cleanup_hdf5() which will free allocated resources though
 * this is somewhat academic, unless you have a lot more work to do.
 *
 */

// Define a data type alias so that other users don't have to hardcode
typedef uint16_t image_t_type;

typedef struct image_t {
    uint16_t *data;
    uint8_t *mask;
    size_t slow;
    size_t fast;
} image_t;

/* data as modules i.e. 3D array */
typedef struct image_modules_t {
    uint16_t *data;
    uint8_t *mask;
    size_t modules;
    size_t slow;
    size_t fast;
} image_modules_t;

/* set up HDF5 files - call at the start */
int setup_hdf5_files(char *master_filename, char *data_filename);

/* clean up at the end */
void cleanup_hdf5();

/* interrogate number / size of images */
size_t get_number_of_images();
size_t get_image_slow();
size_t get_image_fast();

/* read an image, free the image */
image_t get_image(size_t number);
void free_image(image_t image);

/* read an image as modules, free this */
image_modules_t get_image_modules(size_t number);
void free_image_modules(image_modules_t modules);

#endif
