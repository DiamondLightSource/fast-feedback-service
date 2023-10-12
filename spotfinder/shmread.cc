

#include "shmread.hpp"

#include <fmt/core.h>

#include <iostream>
#include <nlohmann/json.hpp>

#include "common.hpp"

using json = nlohmann::json;
using namespace fmt;

SHMRead::SHMRead(const std::string &path) : _base_path(path) {
    // Read the header
    auto header_path = path + "/headers.1";
    std::ifstream f(header_path);
    json data = json::parse(f);

    _num_images =
      data["nimages"].template get<size_t>() * data["ntrigger"].template get<size_t>();
    _image_shape = {
      data["y_pixels_in_detector"].template get<size_t>(),
      data["x_pixels_in_detector"].template get<size_t>(),
    };

    uint8_t bit_depth_image = data["bit_depth_image"].template get<uint8_t>();

    if (bit_depth_image != 16) {
        throw std::runtime_error(format(
          "Can not read image with bit_depth_image={}, only 16", bit_depth_image));
    }
    // Read the mask
    std::vector<int32_t> raw_mask;
    raw_mask.resize(_image_shape[0] * _image_shape[1]);
    std::ifstream f_mask(format("{}/headers.5", _base_path),
                         std::ios::in | std::ios::binary);
    f_mask.read(reinterpret_cast<char *>(raw_mask.data()),
                raw_mask.size() * sizeof(decltype(raw_mask)::value_type));
    // draw_image_data(raw_mask.data(), 0, 0, 20, 20, _image_shape[1], _image_shape[0]);
    _mask.reserve(_image_shape[0] * _image_shape[1]);
    for (auto &v : raw_mask) {
        _mask.push_back(!v);
    }
    // return {destination.data(), static_cast<size_t>(f.gcount())};
}

bool SHMRead::is_image_available(size_t index) {
    return std::filesystem::exists(format("{}/{:06d}.2", _base_path, index));
}

SPAN<uint8_t> SHMRead::get_raw_chunk(size_t index, SPAN<uint8_t> destination) {
    std::ifstream f(format("{}/{:06d}.2", _base_path, index),
                    std::ios::in | std::ios::binary);
    f.read(reinterpret_cast<char *>(destination.data()), destination.size());
    return {destination.data(), static_cast<size_t>(f.gcount())};
}

template <>
bool is_ready_for_read<SHMRead>(const std::string &path) {
    // We need headers.1, and headers.5, to read the metadata
    return std::filesystem::exists(format("{}/headers.1", path))
           && std::filesystem::exists(format("{}/headers.5", path));
}
