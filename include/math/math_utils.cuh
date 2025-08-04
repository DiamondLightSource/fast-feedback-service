/**
 * @file math_utils.cuh
 * @brief Common mathematical utility functions for CUDA device and host code
 * 
 * This header provides commonly used mathematical functions that work in both
 * host and device contexts, with appropriate precision handling.
 */

#pragma once

/*
 * In CUDA 13.0 we could use <cuda/std/numbers> for better precision and type safety.
 * constexpr scalar_t RAD_TO_DEG = 180.0 / cuda::std::numbers::pi_v<scalar_t>;
 */
#include <cmath>

#include "math/device_precision.cuh"

/**
 * @brief Convert radians to degrees
 * @param radians Angle in radians
 * @return Angle in degrees
 */
__device__ __host__ inline scalar_t radians_to_degrees(scalar_t radians) {
    constexpr scalar_t RAD_TO_DEG = 180.0 / static_cast<scalar_t>(M_PI);
    return radians * RAD_TO_DEG;
}

/**
 * @brief Convert degrees to radians
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
__device__ __host__ inline scalar_t degrees_to_radians(scalar_t degrees) {
    constexpr scalar_t DEG_TO_RAD = static_cast<scalar_t>(M_PI) / 180.0;
    return degrees * DEG_TO_RAD;
}