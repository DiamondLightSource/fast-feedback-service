/**
 * @file device_common.hu
 * @brief Common device functions
 */

#ifndef DEVICE_COMMON_H
#define DEVICE_COMMON_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

/* 
 * Kernel radii 🔳
 *
 * Kernel, here, referring to a sliding window of pixels. Such as in
 * convolution or erosion. This is not to be confused with the CUDA
 * kernel, which is a function that runs on the GPU.
 * 
 * One-direction width of kernel. Total kernel span is (R * 2 + 1)
 * The kernel is a square, so the height is the same as the width.
*/
constexpr uint8_t KERNEL_RADIUS = 3;           // 7x7 kernel
constexpr uint8_t KERNEL_RADIUS_EXTENDED = 5;  // 11x11 kernel

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

/*
 * Constants for kernels
 * extern keyword is used to declare a variable that is defined in
 * another file. This links the constant global variable to the
 * kernel_constants variable defined in `spotfinder.cu`
 */
extern __constant__ KernelConstants kernel_constants;

#endif  // DEVICE_COMMON_H