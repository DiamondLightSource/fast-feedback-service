#include <bitshuffle.h>

#include <chrono>
#include <cstdio>
#include <vector>

#include "h5read.h"

int main(int argc, char **argv) {
    auto reader = H5Read(argc, argv);

    size_t n_images = reader.get_number_of_images();
    size_t num_pixels = reader.get_image_slow() * reader.get_image_fast();
    size_t elem_size = reader.get_element_size();

    auto buffer = std::vector<uint8_t>(num_pixels * elem_size);
    auto image = std::vector<uint8_t>(num_pixels * elem_size);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < n_images; j++) {
        auto data = reader.get_raw_chunk(j, buffer);

        // Decompress this
        bshuf_decompress_lz4(
          buffer.data() + 12, image.data(), num_pixels, elem_size, 0);

        printf("Read Image %zu chunk of %zu KBytes\n", j, data.size() / 1024);
    }

    // Work out how long this took and print stats
    auto end_time = std::chrono::high_resolution_clock::now();
    float total_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
        .count();
    printf("\nTook %.2f s (%.0f im/s)\n", total_time, n_images / total_time);
}
