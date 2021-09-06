#ifndef _H5READ_H
#define _H5READ_H

#include <stddef.h>
#include <stdint.h>

/* Basic API specification:
 *
 * Call
 *
 * setup_hdf5_files(master_filename)
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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _h5read_handle h5read_handle;

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
    uint16_t *data;  ///< Module image data in a 3D array block of [module][slow][fast]
    uint8_t *mask;   ///< Image mask, in the same shape as the module data
    size_t modules;  ///< Total number of modules
    size_t slow;     ///< Number of pixels in slow direction per module
    size_t fast;     ///< Number of pixels in fast direction per module
} image_modules_t;

/// Read an h5 file. Returns NULL if failed.
h5read_handle *h5read_open(const char *master_filename);

/// Generate sample data
h5read_handle *h5read_generate_samples();

/// Cleanup and release an h5 file object
void h5read_free(h5read_handle *);

/// Get the number of images in a dataset
size_t h5read_get_number_of_images(h5read_handle *obj);
/// Get the number of image pixels in the slow dimension
size_t h5read_get_image_slow(h5read_handle *obj);
/// Get the number of image pixels in the fast dimension
size_t h5read_get_image_fast(h5read_handle *obj);

/// Read an image from a dataset
image_t *h5read_get_image(h5read_handle *obj, size_t number);
/// Free a previously read image
void h5read_free_image(image_t *image);

/// Read an image from a dataset, split up into modules
image_modules_t *h5read_get_image_modules(h5read_handle *obj, size_t frame_number);
/// Free an image read as modules
void h5read_free_image_modules(image_modules_t *modules);

/// Parse basic command arguments with verbose, filename in form:
///     Usage: <prog> [-h|--help] [-v] [FILE.nxs]
h5read_handle *h5read_parse_standard_args(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif
