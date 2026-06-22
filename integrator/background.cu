/**
 * @file background.cu
 * @brief GPU per-reflection background reduction kernel.
 *
 * Reduces the per-reflection background histograms accumulated by the Kabsch
 * kernel into background estimates, using the single-source model code in
 * integrator/background.hpp so the result matches the baseline CPU path.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "cuda_common.hpp"
#include "integrator.cuh"
#include "integrator/background.cuh"
#include "integrator/background.hpp"

namespace {
constexpr int BACKGROUND_REDUCE_THREADS = 128;
}

/**
 * @brief One thread per reflection: evaluate the background model over that
 *        reflection's histogram slice.
 */
__global__ void background_reduce_kernel(BackgroundModel model,
                                         const uint32_t *d_background_hist,
                                         const uint32_t *d_background_overflow,
                                         size_t num_reflections,
                                         double *d_background_mean,
                                         double *d_background_sum_value,
                                         uint32_t *d_background_count,
                                         uint8_t *d_background_success) {
    const size_t r = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= num_reflections) return;

    ConstHistogramView view;
    view.bins = d_background_hist + r * NUM_BG_BINS;
    view.num_bins = NUM_BG_BINS;
    view.overflow_count = d_background_overflow[r];

    // Total background pixel count for this reflection (num_pixels.background).
    uint32_t total = view.overflow_count;
    for (int v = 0; v < NUM_BG_BINS; ++v) {
        total += view.bins[v];
    }
    d_background_count[r] = total;

    BackgroundResult res;
    switch (model) {
    case BackgroundModel::Constant:
        res = tukey_constant_background(view);
        break;
    case BackgroundModel::Glm:
        // Robust-Poisson GLM constant background (DIALS "glm constant3d"),
        // evaluated over the same histogram view by the shared single-source
        // core so the device matches the baseline.
        res = glm_constant_background(view);
        break;
    }

    d_background_mean[r] = res.mean;
    d_background_sum_value[r] = res.weighted_sum;
    d_background_success[r] = res.valid ? 1u : 0u;
}

void compute_background(BackgroundModel model,
                        const uint32_t *d_background_hist,
                        const uint32_t *d_background_overflow,
                        size_t num_reflections,
                        double *d_background_mean,
                        double *d_background_sum_value,
                        uint32_t *d_background_count,
                        uint8_t *d_background_success,
                        cudaStream_t stream) {
    if (num_reflections == 0) return;

    const unsigned int blocks = static_cast<unsigned int>(
      (num_reflections + BACKGROUND_REDUCE_THREADS - 1) / BACKGROUND_REDUCE_THREADS);

    background_reduce_kernel<<<blocks, BACKGROUND_REDUCE_THREADS, 0, stream>>>(
      model,
      d_background_hist,
      d_background_overflow,
      num_reflections,
      d_background_mean,
      d_background_sum_value,
      d_background_count,
      d_background_success);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(fmt::format("Background reduction launch failed: {}",
                                             cudaGetErrorString(err)));
    }
}
