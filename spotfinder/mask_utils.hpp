/**
 * @file mask_utils.hpp
 * @brief Mask upload and resolution filtering utilities.
 *
 * Provides functions for uploading detector masks to GPU memory
 * and applying resolution-based filtering to masks.
 */
#ifndef MASK_UTILS_HPP
#define MASK_UTILS_HPP

#include <cuda_runtime.h>
#include <fmt/core.h>

#include "cuda_common.hpp"
#include "h5read.h"
#include "kernels/masking.cuh"

/**
 * @brief Copy the mask from a reader into a pitched GPU area.
 *
 * @tparam T Reader type that provides image_shape() and get_mask() methods.
 * @param reader The data reader containing the mask.
 * @return PitchedMalloc<uint8_t> The mask uploaded to GPU memory.
 */
template <typename T>
auto upload_mask(T &reader) -> PitchedMalloc<uint8_t> {
    size_t height = reader.image_shape()[0];
    size_t width = reader.image_shape()[1];

    auto [dev_mask, device_mask_pitch] =
      make_cuda_pitched_malloc<uint8_t>(width, height);

    size_t valid_pixels = 0;
    CudaEvent start, end;
    if (reader.get_mask()) {
        // Count how many valid Mpx in this mask
        for (size_t i = 0; i < width * height; ++i) {
            if (reader.get_mask().value()[i]) {
                valid_pixels += 1;
            }
        }
        start.record();
        cudaMemcpy2DAsync(dev_mask.get(),
                          device_mask_pitch,
                          reader.get_mask()->data(),
                          width,
                          width,
                          height,
                          cudaMemcpyHostToDevice);
        cuda_throw_error();
    } else {
        valid_pixels = width * height;
        start.record();
        cudaMemset(dev_mask.get(), 1, device_mask_pitch * height);
        cuda_throw_error();
    }
    end.record();
    end.synchronize();

    float memcpy_time = end.elapsed_time(start);
    fmt::print("Uploaded mask ({:.2f} Mpx) in {:.2f} ms ({:.1f} GBps)\n",
               static_cast<float>(valid_pixels) / 1e6,
               memcpy_time,
               GBps(memcpy_time, width * height));

    return PitchedMalloc{
      dev_mask,
      width,
      height,
      device_mask_pitch,
    };
}

/**
 * @brief Apply resolution-based filtering to a mask on the GPU.
 *
 * @param mask The mask in GPU memory to modify.
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param wavelength X-ray wavelength in Angstroms.
 * @param detector Detector geometry parameters.
 * @param dmin Minimum resolution in Angstroms (or -1 for no limit).
 * @param dmax Maximum resolution in Angstroms (or -1 for no limit).
 * @param stream CUDA stream for the operation (default: 0).
 */
void apply_resolution_filtering(PitchedMalloc<uint8_t> mask,
                                int width,
                                int height,
                                float wavelength,
                                detector_geometry detector,
                                float dmin,
                                float dmax,
                                cudaStream_t stream = 0);

#endif  // MASK_UTILS_HPP
