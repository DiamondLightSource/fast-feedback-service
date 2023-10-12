#pragma once

#include <cuda_runtime.h>
#include <fmt/core.h>

#include <vector>

#include "h5read.h"

typedef union {
    char b[2];
    short s;
} union_short;

typedef union {
    char b[4];
    int i;
} union_int;

inline void byte_swap_short(char *b) {
    char c;
    c = b[0];
    b[0] = b[1];
    b[1] = c;
    return;
}

inline void byte_swap_int(char *b) {
    char c;
    c = b[0];
    b[0] = b[3];
    b[3] = c;
    c = b[1];
    b[1] = b[2];
    b[2] = c;
    return;
}

inline bool little_endian() {
    int i = 0x1;
    char b = ((union_int *)&i)[0].b[0];
    if (b == 0) {
        return false;
    } else {
        return true;
    }
}

template <typename Tout>
unsigned int cbf_decompress(const char *packed,
                            std::size_t packed_sz,
                            Tout *values,
                            std::size_t values_sz) {
    int current = 0;
    Tout *original = values;
    unsigned int j = 0;
    short s;
    char c;
    int i;
    bool le = little_endian();

    while ((j < packed_sz) && ((values - original) < values_sz)) {
        c = packed[j];
        j += 1;

        if (c != -0x80) {
            current += c;
            *values = current;
            values++;
            continue;
        }

        assert(j + 1 < packed_sz);
        ((union_short *)&s)[0].b[0] = packed[j];
        ((union_short *)&s)[0].b[1] = packed[j + 1];
        j += 2;

        if (!le) {
            byte_swap_short((char *)&s);
        }

        if (s != -0x8000) {
            current += s;
            *values = current;
            values++;
            continue;
        }

        assert(j + 3 < packed_sz);
        ((union_int *)&i)[0].b[0] = packed[j];
        ((union_int *)&i)[0].b[1] = packed[j + 1];
        ((union_int *)&i)[0].b[2] = packed[j + 2];
        ((union_int *)&i)[0].b[3] = packed[j + 3];
        j += 4;

        if (!le) {
            byte_swap_int((char *)&i);
        }

        current += i;
        *values = current;
        values++;
    }

    return values - original;
}

template <typename Tout>
void decompress_byte_offset(const SPAN<uint8_t> in, SPAN<Tout> out) {
    cbf_decompress(reinterpret_cast<const char *>(in.data()),
                   in.size_bytes(),
                   out.data(),
                   out.size());
}

class CBFRead : public Reader {
  private:
    size_t _num_images;
    size_t _first_index;
    std::array<size_t, 2> _image_shape;
    const std::string _template_path;
    std::vector<uint8_t> _mask;

  public:
    CBFRead(const std::string &templatestr, size_t num_images, size_t first_index);

    bool is_image_available(size_t index);

    SPAN<uint8_t> get_raw_chunk(size_t index, SPAN<uint8_t> destination);

    ChunkCompression get_raw_chunk_compression() {
        return Reader::ChunkCompression::BYTE_OFFSET_32;
    }
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

template <typename Tout>
void decompress_byte_offset(const SPAN<uint8_t> in, SPAN<Tout> out);