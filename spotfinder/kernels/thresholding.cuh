#pragma once

#include "h5read.h"

using pixel_t = H5Read::image_type;

__global__ void dispersion(pixel_t __restrict__ *image,
                                         uint8_t __restrict__ *mask,
                                         uint8_t __restrict__ *result_mask,
                                         size_t image_pitch,
                                         size_t mask_pitch,
                                         size_t result_pitch,
                                         int width,
                                         int height,
                                         pixel_t max_valid_pixel_value,
                                         uint8_t kernel_width,
                                         uint8_t kernel_height,
                                         uint8_t min_count,
                                         float n_sig_b,
                                         float n_sig_s);

__global__ void dispersion_extended_first_pass(pixel_t __restrict__ *image,
                                                    uint8_t __restrict__ *mask,
                                                    uint8_t __restrict__ *result_mask,
                                                    size_t image_pitch,
                                                    size_t mask_pitch,
                                                    size_t result_pitch,
                                                    int width,
                                                    int height,
                                                    pixel_t max_valid_pixel_value,
                                                    uint8_t kernel_width,
                                                    uint8_t kernel_height,
                                                    uint8_t min_count,
                                                    float n_sig_b,
                                                    float n_sig_s);

__global__ void dispersion_extended_second_pass(pixel_t __restrict__ *image,
                                               uint8_t __restrict__ *mask,
                                               uint8_t __restrict__ *dispersion_mask,
                                               uint8_t __restrict__ *result_mask,
                                               size_t image_pitch,
                                               size_t mask_pitch,
                                               size_t dispersion_mask_pitch,
                                               size_t result_mask_pitch,
                                               int width,
                                               int height,
                                               pixel_t max_valid_pixel_value,
                                               uint8_t kernel_width,
                                               uint8_t kernel_height,
                                               float n_sig_s,
                                               float threshold);
