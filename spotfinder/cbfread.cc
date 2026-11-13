#include "cbfread.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string_view>

#include "common.hpp"

namespace {

/// Marks the start of the binary data section in a CBF file
constexpr std::array<uint8_t, 4> BINARY_MARKER = {0x0c, 0x1a, 0x04, 0xd5};

/// Read an entire file into a byte array
auto read_file(const std::string &filename) -> std::vector<uint8_t> {
    std::ifstream f(filename, std::ios::in | std::ios::binary);
    if (!f) {
        throw std::runtime_error(fmt::format("Could not open CBF file {}", filename));
    }
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

/// Find the binary data section, or throw.
///
/// Uses the X-Binary-Size MIME header, if present, to bound the length.
auto find_binary_data(std::vector<uint8_t> &data, const std::string &filename)
  -> std::span<uint8_t> {
    auto it =
      std::search(data.begin(), data.end(), BINARY_MARKER.begin(), BINARY_MARKER.end());
    if (it == data.end()) {
        throw std::runtime_error(
          fmt::format("Could not find binary section in CBF file {}", filename));
    }
    size_t start = std::distance(data.begin(), it) + BINARY_MARKER.size();
    size_t size = data.size() - start;

    static const std::regex re_size(R"(X-Binary-Size:\s*([0-9]+))");
    std::string header(data.begin(), it);
    std::smatch m;
    if (std::regex_search(header, m, re_size)) {
        size = std::min<size_t>(size, std::stoull(m[1]));
    }
    return {data.data() + start, size};
}

/// Expand a '#' template to a specific image number
auto expand_template(const std::string &template_path, size_t index) -> std::string {
    auto first = template_path.find('#');
    auto last = template_path.rfind('#');
    return fmt::format("{}{:0{}d}{}",
                       template_path.substr(0, first),
                       index,
                       last - first + 1,
                       template_path.substr(last + 1));
}

auto parse_float(const std::string &s) -> float {
    return std::stof(s);
}

}  // namespace

std::string CBFRead::filename_for(const std::string &path, size_t index) {
    return is_template(path) ? expand_template(path, index) : path;
}

CBFRead::CBFRead(const std::string &path, size_t num_images, size_t first_index)
    : _path(path),
      _is_template(is_template(path)),
      _num_images(_is_template ? num_images : 1) {
    auto first_filename = filename_for(first_index);
    auto file_data = read_file(first_filename);
    auto binary_data = find_binary_data(file_data, first_filename);

    // Parse the text header. This is the (mini)CBF header, and the MIME
    // header immediately before the binary section.
    std::string header(file_data.data(), binary_data.data());
    std::optional<size_t> fast, slow;

    // Matches e.g. "# Beam_xy (1263.40, 1318.60) pixels"
    const std::regex re_beam_xy(
      R"(Beam_xy\s*\(\s*([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)\s*\))");
    const std::regex re_pixel_size(
      R"(Pixel_size\s+([-0-9.eE+]+)\s*m\s*x\s*([-0-9.eE+]+)\s*m)");
    const std::regex re_wavelength(R"(Wavelength\s+([-0-9.eE+]+)\s*A)");
    const std::regex re_distance(R"(Detector_distance\s+([-0-9.eE+]+)\s*m)");
    const std::regex re_cutoff(R"(Count_cutoff\s+([0-9]+))");
    const std::regex re_fast(R"(X-Binary-Size-Fastest-Dimension:\s*([0-9]+))");
    const std::regex re_slow(R"(X-Binary-Size-Second-Dimension:\s*([0-9]+))");

    std::smatch m;
    if (std::regex_search(header, m, re_fast)) fast = std::stoul(m[1]);
    if (std::regex_search(header, m, re_slow)) slow = std::stoul(m[1]);
    if (!fast || !slow) {
        throw std::runtime_error(
          fmt::format("Could not read image dimensions from CBF {}", first_filename));
    }
    _image_shape = {*slow, *fast};

    if (std::regex_search(header, m, re_wavelength)) {
        _wavelength = parse_float(m[1]);
    }
    if (std::regex_search(header, m, re_distance)) {
        _distance = parse_float(m[1]);
    }
    if (std::regex_search(header, m, re_pixel_size)) {
        // Header is (x, y); the reader interface is (y, x)
        _pixel_size = {{parse_float(m[2]), parse_float(m[1])}};
    }
    if (std::regex_search(header, m, re_beam_xy)) {
        _beam_center = {{parse_float(m[2]), parse_float(m[1])}};
    }
    if (std::regex_search(header, m, re_cutoff)) {
        _count_cutoff = std::stoll(m[1]);
    }

    // Generate the mask from the first image: Negative pixels are invalid
    const size_t num_pixels = _image_shape[0] * _image_shape[1];
    std::vector<int32_t> image_data(num_pixels);
    decompress_byte_offset<int32_t>(binary_data, {image_data.data(), num_pixels});
    _mask.resize(num_pixels);
    for (size_t px = 0; px < num_pixels; ++px) {
        _mask[px] = image_data[px] >= 0;
    }
}

bool CBFRead::is_image_available(size_t index) {
    return std::filesystem::exists(filename_for(index));
}

std::span<uint8_t> CBFRead::get_raw_chunk(size_t index,
                                          std::span<uint8_t> destination) {
    auto filename = filename_for(index);
    auto file_data = read_file(filename);
    auto binary_data = find_binary_data(file_data, filename);

    if (binary_data.size() > destination.size_bytes()) {
        throw std::runtime_error(fmt::format(
          "CBF {} compressed data ({} B) is larger than the destination buffer ({} B)",
          filename,
          binary_data.size(),
          destination.size_bytes()));
    }
    std::copy(binary_data.begin(), binary_data.end(), destination.begin());
    return {destination.data(), binary_data.size()};
}
