#pragma once

#include "h5read.h"

__global__ void dispersion(pixel_t __restrict__ *image,
                           uint8_t __restrict__ *mask,
                           uint8_t __restrict__ *result_mask);

__global__ void dispersion_extended_first_pass(pixel_t __restrict__ *image,
                                               uint8_t __restrict__ *mask,
                                               uint8_t __restrict__ *result_mask);

__global__ void dispersion_extended_second_pass(pixel_t __restrict__ *image,
                                                uint8_t __restrict__ *mask,
                                                uint8_t __restrict__ *dispersion_mask,
                                                uint8_t __restrict__ *result_mask,
                                                size_t dispersion_mask_pitch);
