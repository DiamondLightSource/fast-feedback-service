/**
 * @file device_common.hu
 * @brief Common device functions
 */

#ifndef DEVICE_COMMON_H
#define DEVICE_COMMON_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

/**
 * @brief Struct to act as a global container for constant values
 * necessary for spotfinding
 * 
 * @note This struct is intended to be copied to the device's
 * constant memory before any kernel calls
 */
struct KernelConstants {
    size_t image_pitch;           // Pitch of the image
    size_t mask_pitch;            // Pitch of the mask
    size_t result_pitch;          // Pitch of the result
    ushort width;                 // Width of the image
    ushort height;                // Height of the image
    float max_valid_pixel_value;  // Maximum valid pixel value
    uint8_t min_count;            // Minimum number of pixels in a spot
    float n_sig_b;                // Number of standard deviations for background
    float n_sig_s;                // Number of standard deviations for signal
};

#endif  // DEVICE_COMMON_H