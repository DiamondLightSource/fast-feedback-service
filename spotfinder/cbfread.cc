
#include "cbfread.hpp"

#include <fmt/core.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string_view>

#include "bitshuffle.h"
#include "common.hpp"
using namespace fmt;

// const std::string BINARY_MARKER = "--CIF-BINARY-FORMAT-SECTION--";
const std::string BINARY_MARKER = "\x0c\x1a\x04\xd5";

auto expand_template(const std::string &template_path, size_t index) -> std::string {
    std::string prefix = template_path.substr(0, template_path.find("#"));
    std::string suffix = template_path.substr(template_path.rfind("#") + 1);
    int template_length = template_path.length() - prefix.length() - suffix.length();

    return format("{}{:0{}d}{}", prefix, index, template_length, suffix);
}

// template <>
// void decompress_byte_offset(const SPAN<uint8_t> in, SPAN<uint16_t> out);

/// Splits a header line by middle-space and returns the second value
auto get_value_contents(const std::string &input) -> std::string {
    auto start = input.find_first_not_of(" #");
    auto end = input.find_last_not_of(" #\r");
    auto strimmed = input.substr(start, end - start + 1);

    return strimmed.substr(strimmed.find(" ") + 1);
}

CBFRead::CBFRead(const std::string &templatestr, size_t num_images, size_t first_index)
    : _num_images(num_images), _first_index(first_index), _template_path(templatestr) {
    if (first_index > 1) {
        print("Error: Can only handle CBF start index of 0 or 1\n");
        std::exit(1);
    }
    // We must have our first file, as we read this for mask and metadata
    assert(std::filesystem::exists(expand_template(templatestr, _first_index)));

    {
        std::ifstream f(expand_template(templatestr, first_index), std::ios::in);
        std::string line;
        // Fast-forward to the binary section
        int read_values = 0;
        while (read_values < 2) {
            std::getline(f, line);
            if (line.starts_with("X-Binary-Size-Fastest-Dimension")) {
                _image_shape[1] = std::stoi(get_value_contents(line));
                read_values += 1;
            } else if (line.starts_with("X-Binary-Size-Second-Dimension")) {
                _image_shape[0] = std::stoi(get_value_contents(line));
                read_values += 1;
            }
        }
    }
    // Read the data for the first image to generate a mask
    const size_t num_pixels = _image_shape[0] * _image_shape[1];
    // CBF files are compressed 32-bit, so need more storage
    auto compressed_data_buffer = std::make_unique<uint8_t[]>(num_pixels * 4);
    get_raw_chunk(0, {compressed_data_buffer.get(), num_pixels * 4});
    for (int i = 0; i < 32; ++i) {
        print("{:02x} ", compressed_data_buffer[i]);
    }
    print("\n");
    // auto compressed_data =
    //   get_raw_chunk(0, {compressed_data_buffer.get(), 4 * num_pixels});
    auto image_data = std::make_unique<int16_t[]>(num_pixels);

    decompress_byte_offset<decltype(image_data)::element_type>(
      {compressed_data_buffer.get(), num_pixels * 4}, {image_data.get(), num_pixels});
    _mask.reserve(num_pixels);
    for (size_t px = 0; px < num_pixels; ++px) {
        _mask.push_back(image_data[px] < 0);
    }
    // Go through entire image, using mask of "Everything negative"
    draw_image_data(image_data.get(), 0, 190, 30, 30, _image_shape[1], _image_shape[0]);
    draw_image_data(_mask.data(), 0, 190, 30, 30, _image_shape[1], _image_shape[0]);
}

bool CBFRead::is_image_available(size_t index) {
    return std::filesystem::exists(
      expand_template(_template_path, index + _first_index));
}

SPAN<uint8_t> CBFRead::get_raw_chunk(size_t index, SPAN<uint8_t> destination) {
    auto filename = expand_template(_template_path, index + _first_index);

    // std::vector<char> file_data;
    std::string file_data;
    file_data.reserve(std::filesystem::file_size(filename));
    // file_data.reserve);

    auto f = std::ifstream(filename, std::ios::in | std::ios::binary);
    std::array<char, 4096> read_buffer;

    // Read until we find the binary marker
    do {
        f.read(read_buffer.data(), read_buffer.size());
        file_data.append(read_buffer.data(), f.gcount());
    } while (f.gcount());
    size_t data_start = file_data.find(BINARY_MARKER) + BINARY_MARKER.length();
    // auto byte_data = std::string_view

    //   reinterpret_cast<uint8_t *>(file_data.data()) + data_start,
    //   file_data.length() - data_start};

    assert(destination.size_bytes() >= (file_data.length() - data_start));
    // auto resized_span =
    //   SPAN<uint8_t>(destination.begin(), file_data.length() - data_start);
    std::copy(file_data.begin() + data_start, file_data.end(), destination.begin());

    for (size_t i = data_start; i < file_data.length(); ++i) {
        destination[i - data_start] = file_data[i];
    }

    return {destination.data(), file_data.length() - data_start};
}

template <>
bool is_ready_for_read<CBFRead>(const std::string &path) {
    // Wait for the first (or second if starting from 0) image to exists
    return std::filesystem::exists(expand_template(path, 1));
}
