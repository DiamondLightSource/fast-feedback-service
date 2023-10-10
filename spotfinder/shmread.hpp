#pragma once

#include <cuda_runtime.h>
#include <fmt/core.h>

#include <vector>

#include "h5read.h"

class SHMRead : public Reader {
  private:
    size_t _num_images;
    std::array<size_t, 2> _image_shape;
    const std::string _base_path;
    std::vector<uint8_t> _mask;

  public:
    SHMRead(const std::string &path);

    bool is_image_available(size_t index);

    SPAN<uint8_t> get_raw_chunk(size_t index, SPAN<uint8_t> destination);

    size_t get_number_of_images() const {
        return _num_images;
    }
    std::array<size_t, 2> image_shape() const {
        return _image_shape;
    };
    std::optional<SPAN<const uint8_t>> get_mask() const {
        return {{_mask.data(), _mask.size()}};
    }
};

template <>
bool is_ready_for_read<SHMRead>(const std::string &path);