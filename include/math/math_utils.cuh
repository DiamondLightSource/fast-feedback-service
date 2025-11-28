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

#ifdef __CUDACC__
#include "math/device_precision.cuh"
#define DEVICE_HOST __device__ __host__
using scalar_type = scalar_t;
#else
#define DEVICE_HOST inline
using scalar_type = double;
#endif

/**
 * @brief Convert radians to degrees
 * @param radians Angle in radians
 * @return Angle in degrees
 */
DEVICE_HOST scalar_type radians_to_degrees(scalar_type radians) {
    constexpr scalar_type RAD_TO_DEG = 180.0 / static_cast<scalar_type>(M_PI);
    return radians * RAD_TO_DEG;
}

/**
 * @brief Convert degrees to radians
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
DEVICE_HOST scalar_type degrees_to_radians(scalar_type degrees) {
    constexpr scalar_type DEG_TO_RAD = static_cast<scalar_type>(M_PI) / 180.0;
    return degrees * DEG_TO_RAD;
}

#undef DEVICE_HOST