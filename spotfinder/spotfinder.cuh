#ifndef SPOTFINDER_H
#define SPOTFINDER_H

#include <builtin_types.h>

#include "common.hpp"
#include "cuda_common.hpp"
#include "h5read.h"

#define VALID_PIXEL 1
#define MASKED_PIXEL 0

using pixel_t = H5Read::image_type;

void call_do_spotfinding_dispersion(dim3 blocks,
                                    dim3 threads,
                                    size_t shared_memory,
                                    cudaStream_t stream,
                                    PitchedMalloc<pixel_t> &image,
                                    PitchedMalloc<uint8_t> &mask,
                                    int width,
                                    int height,
                                    pixel_t max_valid_pixel_value,
                                    PitchedMalloc<uint8_t> *result_strong,
                                    uint8_t min_count = 3,
                                    float nsig_b = 6.0f,
                                    float nsig_s = 3.0f);

void call_do_spotfinding_extended(dim3 blocks,
                                  dim3 threads,
                                  size_t shared_memory,
                                  cudaStream_t stream,
                                  PitchedMalloc<pixel_t> &image,
                                  PitchedMalloc<uint8_t> &mask,
                                  int width,
                                  int height,
                                  pixel_t max_valid_pixel_value,
                                  PitchedMalloc<uint8_t> *result_strong,
                                  bool do_writeout = false,
                                  uint8_t min_count = 3,
                                  float nsig_b = 6.0f,
                                  float nsig_s = 3.0f,
                                  float threshold = 0.0f);

#endif