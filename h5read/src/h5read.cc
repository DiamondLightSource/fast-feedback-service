#include <h5read.h>

#include <cassert>
#include <iostream>

using namespace std;

auto h5read_freeer = [](h5read_handle *h) { h5read_free(h); };
auto h5read_image_freeer = [](image_t *h) { h5read_free_image(h); };
auto h5read_image_modules_freeer = [](image_modules_t *h) {
    h5read_free_image_modules(h);
};

H5Read::H5Read() {
    _handle = std::shared_ptr<h5read_handle>(h5read_generate_samples(), h5read_freeer);
}

H5Read::H5Read(const std::string &filename) {
#ifdef HAVE_HDF5
    auto obj = h5read_open(filename.c_str());
    if (obj == nullptr) throw std::runtime_error("Could not open Nexus file");
    _handle = std::shared_ptr<h5read_handle>(obj, h5read_freeer);
#else
    throw std::runtime_error(
      "h5read built without HDF5 support; cannot open nexus file");
#endif
}

H5Read::H5Read(int argc, char **argv) noexcept {
    _handle = std::shared_ptr<h5read_handle>(h5read_parse_standard_args(argc, argv),
                                             h5read_freeer);
}

Image::Image(std::shared_ptr<h5read_handle> handle, size_t i) noexcept
    : _handle(handle),
      _image{std::shared_ptr<image_t>(h5read_get_image(_handle.get(), i),
                                      h5read_image_freeer)},
      mask{_image->mask, _image->slow * _image->fast},
      slow{_image->slow},
      fast{_image->fast},
      dtype{_image->dtype} {
    // Currently h5read_get_image guarantees that it will never return invalid
    assert(_image);
}

ImageModules::ImageModules(std::shared_ptr<h5read_handle> handle, size_t i) noexcept
    : _handle{handle},
      _modules{
        std::shared_ptr<image_modules_t>(h5read_get_image_modules(_handle.get(), i),
                                         h5read_image_modules_freeer)},
      _modules_masks{_modules->modules},
      mask{_modules->mask, _modules->slow * _modules->fast * _modules->modules},
      n_modules{_modules->modules},
      slow{_modules->slow},
      fast{_modules->fast},
      dtype{_modules->dtype},
      masks{&_modules_masks.front(), _modules->modules} {
    // Build the mask per-module lookups
    for (size_t j = 0; j < _modules->modules; ++j) {
        _modules_masks[j] = std::span{_modules->mask + slow * fast * j, slow * fast};
    }
}
