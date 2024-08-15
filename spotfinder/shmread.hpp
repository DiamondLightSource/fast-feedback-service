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
    std::array<image_t_type, 2> _trusted_range;

  public:
    SHMRead(const std::string &path);

    bool is_image_available(size_t index);

    std::span<uint8_t> get_raw_chunk(size_t index, std::span<uint8_t> destination);

    virtual auto get_raw_chunk_compression() -> ChunkCompression {
        return Reader::ChunkCompression::BITSHUFFLE_LZ4;
    }

    size_t get_number_of_images() const {
        return _num_images;
    }
    std::array<size_t, 2> image_shape() const {
        return _image_shape;
    };
    std::optional<std::span<const uint8_t>> get_mask() const {
        return {{_mask.data(), _mask.size()}};
    }
    virtual std::array<image_t_type, 2> get_trusted_range() const {
        return _trusted_range;
    }
};

template <>
bool is_ready_for_read<SHMRead>(const std::string &path);