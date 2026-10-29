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
//
// A foreground sum is bounded by the pixel maximum times the shoebox
// foreground count, so the accumulator width follows the pixel width: 16-bit
// sums fit uint32_t and its single-instruction atomicAdd, 32-bit sums need 64
// bits.
//
// NOTE: If pixel_t can be negative (pedestal-subtracted data), change to a
// signed type of the same width.
// NOTE: If pixel values are non-integer (e.g. gain-corrected floats), revert to:
//     using accumulator_t = scalar_t;
#ifdef PIXEL_DATA_32BIT
using accumulator_t = unsigned long long;
#else
using accumulator_t = uint32_t;
#endif