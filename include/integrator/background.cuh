/**
 * @file background.cuh
 * @brief GPU per-reflection background reduction.
 *
 * After the Kabsch kernel has accumulated a per-reflection background
 * histogram on the device, this reduces each reflection's histogram into a
 * background estimate (mean, inlier weighted sum, pixel count, success) using
 * the selected BackgroundModel. The reduction runs entirely on the device and
 * reuses the single-source model code in integrator/background.hpp, so it
 * produces the same result as the baseline CPU path.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "integrator/background.hpp"

/**
 * @brief Reduce per-reflection background histograms into background estimates.
 *
 * One thread handles one reflection: it views its slice of the histogram
 * (NUM_BG_BINS bins plus the overflow tail) and evaluates the chosen model.
 *
 * @param model Background model to apply (Constant = Tukey; Glm = robust-Poisson GLM)
 * @param d_background_hist Device histogram, num_reflections * NUM_BG_BINS bins
 * @param d_background_overflow Device per-reflection high-tail counts
 * @param num_reflections Number of reflections
 * @param d_background_mean Output: per-reflection background level (per pixel)
 * @param d_background_sum_value Output: per-reflection inlier weighted sum (background.sum.value)
 * @param d_background_count Output: per-reflection total background pixel count
 * @param d_background_success Output: per-reflection success flag (1 = estimate valid)
 * @param stream CUDA stream for async execution
 */
void compute_background(BackgroundModel model,
                        const uint32_t *d_background_hist,
                        const uint32_t *d_background_overflow,
                        size_t num_reflections,
                        double *d_background_mean,
                        double *d_background_sum_value,
                        uint32_t *d_background_count,
                        uint8_t *d_background_success,
                        cudaStream_t stream);
