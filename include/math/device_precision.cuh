/**
 * @file device_precision.cuh
 * @brief Centralized precision configuration for CUDA device code
 * 
 * This header provides unified type aliases and precision-aware
 * function macros that automatically select appropriate precision based
 * on the USE_DOUBLE_PRECISION compile flag.
 */

#pragma once

#include <cuda_runtime.h>

// Precision-dependent configuration
#ifdef USE_DOUBLE_PRECISION
using scalar_t = double;

// Double precision math functions
#define CUDA_MIN fmin
#define CUDA_MAX fmax
#define CUDA_FLOOR floor
#define CUDA_CEIL ceil
#define CUDA_ABS fabs
#define CUDA_SQRT sqrt
#define CUDA_EXP exp
#else
using scalar_t = float;

// Single precision math functions
#define CUDA_MIN fminf
#define CUDA_MAX fmaxf
#define CUDA_FLOOR floorf
#define CUDA_CEIL ceilf
#define CUDA_ABS fabsf
#define CUDA_SQRT sqrtf
#define CUDA_EXP expf
#endif

// Accumulator type for summation/reduction operations.
// Integer accumulation is used because:
// 1. atomicAdd on uint32_t is a single native hardware instruction (fastest atomic)
// 2. Maximum possible sum is well within uint32_t range (~20M << 4.3G)
// 3. Final reduction to double is performed on the host
//
// NOTE: If pixel_t can be negative (pedestal-subtracted data), change to int32_t.
// NOTE: If pixel values are non-integer (e.g. gain-corrected floats), revert to:
//     using accumulator_t = scalar_t;
using accumulator_t = uint32_t;