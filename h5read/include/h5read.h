#ifndef _H5READ_H
#define _H5READ_H

#include <stddef.h>
#include <stdint.h>

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

#include <array>
#include <memory>
#include <string>
#include <vector>

// We might be on an implementation that doesn't have <span>, so use a backport
#ifdef USE_SPAN_BACKPORT
#include "span.hpp"
#define SPAN tcb::span
#else
#include <span>
#define SPAN std::span
#endif

class Image {
  private:
    std::shared_ptr<h5read_handle> _handle;
    std::shared_ptr<image_t> _image;

  public:
    Image(std::shared_ptr<h5read_handle> reader, size_t i) noexcept;

    const SPAN<image_t_type> data;
    const SPAN<uint8_t> mask;
    const size_t slow;
    const size_t fast;
};

class ImageModules {
  private:
    std::shared_ptr<h5read_handle> _handle;
    std::shared_ptr<image_modules_t> _modules;
    // Mutable vectors to point to the appropriate parts of the module data
    std::vector<SPAN<image_t_type>> _modules_data;
    std::vector<SPAN<uint8_t>> _modules_masks;

  public:
    ImageModules(std::shared_ptr<h5read_handle> handle, size_t i) noexcept;

    const SPAN<image_t_type> data;
    const SPAN<uint8_t> mask;

    const size_t n_modules;  ///< Number of modules
    const size_t slow;       ///< Height of a module, in pixels
    const size_t fast;       ///< Width of a module, in pixels
    /// Convenience lookup to get data for a particular module
    const SPAN<SPAN<image_t_type>> modules;
    /// Convenience lookup to get mask for a particular module
    const SPAN<SPAN<uint8_t>> masks;
};

// Declare a C++ "object" version so we don't have to keep track of allocations
class H5Read {
  public:
    /// Create a reader using generated sample data
    H5Read();
    /// Create a reader from a Nexus file
    H5Read(const std::string &filename);
    /// Create a reader by parsing command arguments
    H5Read(int argc, char **argv) noexcept;

    Image get_image(size_t index) {
        return Image(_handle, index);
    }

    ImageModules get_image_modules(size_t index) {
        return ImageModules(_handle, index);
    }

    /// Get the total number of image frames
    size_t get_number_of_images() {
        return h5read_get_number_of_images(_handle.get());
    }
    /// Get the number of pixels in the slow dimension
    size_t get_image_slow() {
        return h5read_get_image_slow(_handle.get());
    }
    /// Get the number of pixels in the fast dimension
    size_t get_image_fast() {
        return h5read_get_image_fast(_handle.get());
    }
    /// Get the image shape, in (slow, fast) pixels
    std::array<size_t, 2> image_shape() {
        return {get_image_slow(), get_image_fast()};
    }

  protected:
    std::shared_ptr<h5read_handle> _handle;
};
#endif

#endif
