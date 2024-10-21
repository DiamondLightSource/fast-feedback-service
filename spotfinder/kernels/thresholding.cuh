#pragma once

#include "../spotfinder.cuh"

__global__ void compute_threshold_kernel(pixel_t *image,
                                         uint8_t *mask,
                                         uint8_t *result_mask,
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

__global__ void compute_dispersion_threshold_kernel(pixel_t *image,
                                                    uint8_t *mask,
                                                    uint8_t *result_mask,
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

__global__ void compute_final_threshold_kernel(pixel_t *image,
                                               uint8_t *mask,
                                               uint8_t *dispersion_mask,
                                               uint8_t *result_mask,
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
