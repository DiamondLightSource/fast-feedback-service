/**
 * @file device_precision.cuh
 * @brief Centralized precision configuration for CUDA device code
 * 
 * This header provides unified type aliases and precision-aware
 * function macros that select the working precision based on the
 * USE_REDUCED_PRECISION compile flag. Double is the default (bit-exact
 * against the CPU baseline); defining USE_REDUCED_PRECISION switches to
 * single precision, which is faster on FP64-throttled GPUs but drifts the
 * pixel classification off the double result.
 */

#pragma once

#include <cuda_runtime.h>

// Precision-dependent configuration. Double is the default; defining
// USE_REDUCED_PRECISION switches to single.
#ifdef USE_REDUCED_PRECISION
using scalar_t = float;

// Single precision math functions
#define CUDA_MIN fminf
#define CUDA_MAX fmaxf
#define CUDA_FLOOR floorf
#define CUDA_CEIL ceilf
#define CUDA_ABS fabsf
#define CUDA_SQRT sqrtf
#define CUDA_EXP expf
#else
using scalar_t = double;

// Double precision math functions
#define CUDA_MIN fmin
#define CUDA_MAX fmax
#define CUDA_FLOOR floor
#define CUDA_CEIL ceil
#define CUDA_ABS fabs
#define CUDA_SQRT sqrt
#define CUDA_EXP exp
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