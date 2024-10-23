#pragma once

__global__ void erosion_kernel(uint8_t __restrict__ *dispersion_mask,
                               uint8_t __restrict__ *erosion_mask,
                               uint8_t __restrict__ *mask,
                               size_t dispersion_mask_pitch,
                               size_t erosion_mask_pitch,
                               size_t mask_pitch,
                               int width,
                               int height,
                               uint8_t radius);