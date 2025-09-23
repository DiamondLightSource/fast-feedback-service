#pragma once

__global__ void erosion(uint8_t __restrict__ *dispersion_mask_ptr,
                        uint8_t __restrict__ *erosion_mask_ptr,
                        uint8_t __restrict__ *mask_ptr,
                        size_t dispersion_mask_pitch,
                        size_t erosion_mask_pitch,
                        size_t mask_pitch,
                        uint8_t radius);