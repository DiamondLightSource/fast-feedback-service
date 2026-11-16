#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "h5read.h"

/// Decompress CBF "byte_offset" packed data into an output array.
///
/// Values are accumulated as signed 32-bit and then converted to Tout,
/// so negative (masked) pixels wrap when Tout is unsigned. Callers rely
/// on the reader mask, not the pixel values, to exclude these.
///
/// @returns The number of values written.
template <typename Tout>
size_t cbf_decompress(const uint8_t *packed,
                      std::size_t packed_sz,
                      Tout *values,
                      std::size_t values_sz) {
    int32_t current = 0;
    size_t j = 0;
    size_t n = 0;

    // The format is little-endian, irrespective of host
    auto read_le16 = [&](size_t at) -> int16_t {
        return static_cast<int16_t>(packed[at] | (packed[at + 1] << 8));
    };
    auto read_le32 = [&](size_t at) -> int32_t {
        return static_cast<int32_t>(static_cast<uint32_t>(packed[at])
                                    | (static_cast<uint32_t>(packed[at + 1]) << 8)
                                    | (static_cast<uint32_t>(packed[at + 2]) << 16)
                                    | (static_cast<uint32_t>(packed[at + 3]) << 24));
    };

    while (j < packed_sz && n < values_sz) {
        int8_t delta8 = static_cast<int8_t>(packed[j]);
        j += 1;
        if (delta8 != -0x80) {
            current += delta8;
        } else {
            if (j + 2 > packed_sz) break;
            int16_t delta16 = read_le16(j);
            j += 2;
            if (delta16 != -0x8000) {
                current += delta16;
            } else {
                if (j + 4 > packed_sz) break;
                current += read_le32(j);
                j += 4;
            }
        }
        values[n++] = static_cast<Tout>(current);
    }
    return n;
}

template <typename Tout>
void decompress_byte_offset(const std::span<uint8_t> in, std::span<Tout> out) {
    size_t written = cbf_decompress(in.data(), in.size_bytes(), out.data(), out.size());
    if (written != out.size()) {
        throw std::runtime_error(
          "CBF byte_offset data decompressed to fewer pixels than expected");
    }
}

/// Reader for (PILATUS-style, "miniCBF") byte-offset compressed CBF images.
///
/// The path is either:
///   - A single CBF file. This reader then has exactly one image, and
///     every image index refers to that file.
///   - A template, with a run of '#' characters standing for the image
///     number, e.g. "image_#####.cbf". Image index N reads the file whose
///     number is N (the caller applies any start offset).
///
/// Metadata (shape, wavelength, geometry, mask) is read from the first
/// image. Pixels with negative values in that image (module gaps, bad
/// pixels) are masked, as DIALS does for PILATUS detectors.
class CBFRead : public Reader {
  private:
    const std::string _path;
    const bool _is_template;
    size_t _num_images;
    std::array<size_t, 2> _image_shape;
    std::vector<uint8_t> _mask;
    int64_t _count_cutoff = std::numeric_limits<int32_t>::max();
    std::optional<float> _wavelength;
    std::optional<std::array<float, 2>> _pixel_size;
    std::optional<std::array<float, 2>> _beam_center;
    std::optional<float> _distance;

  public:
    /// @param path        A CBF filename, or a '#' template
    /// @param num_images  Number of images (ignored for a single file)
    /// @param first_index File number of the first image (template only)
    CBFRead(const std::string &path, size_t num_images, size_t first_index);

    /// Whether a path names a '#' template rather than a single file
    static bool is_template(const std::string &path) {
        return path.find('#') != std::string::npos;
    }
    /// The filename that holds a specific image index, for a file or template
    static std::string filename_for(const std::string &path, size_t index);
    /// The filename that holds a specific image index
    std::string filename_for(size_t index) const {
        return filename_for(_path, index);
    }

    h5read_dtype get_dtype() const override {
        // CBF data is natively 32-bit, so must use the 32-bit spotfinder
        return H5READ_DTYPE_UINT32;
    }
    bool is_image_available(size_t index) override;

    std::span<uint8_t> get_raw_chunk(size_t index,
                                     std::span<uint8_t> destination) override;

    ChunkCompression get_raw_chunk_compression() override {
        return Reader::ChunkCompression::BYTE_OFFSET_32;
    }
    size_t get_number_of_images() const override {
        return _num_images;
    }
    std::array<size_t, 2> image_shape() const override {
        return _image_shape;
    };
    std::optional<std::span<const uint8_t>> get_mask() const override {
        return {{_mask.data(), _mask.size()}};
    }
    std::array<int64_t, 2> get_trusted_range() const override {
        return {0, _count_cutoff};
    }
    std::optional<float> get_wavelength() const override {
        return _wavelength;
    }
    std::optional<std::array<float, 2>> get_pixel_size() const override {
        return _pixel_size;
    };
    std::optional<std::array<float, 2>> get_beam_center() const override {
        return _beam_center;
    };
    std::optional<float> get_detector_distance() const override {
        return _distance;
    };
    std::array<float, 2> get_oscillation() const override {
        // Treated as stills
        return {0, 0};
    };
};
