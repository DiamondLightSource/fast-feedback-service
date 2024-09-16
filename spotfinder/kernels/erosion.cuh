#pragma once

__global__ void erosion_kernel(const uint8_t __restrict__ *mask,
                               uint8_t __restrict__ *erosion_mask,
                               size_t mask_pitch,
                               int width,
                               int height,
                               int radius);