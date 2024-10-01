#pragma once

__global__ void dilation_kernel(uint8_t __restrict__ *mask,
                               size_t mask_pitch,
                               int width,
                               int height,
                               int radius);