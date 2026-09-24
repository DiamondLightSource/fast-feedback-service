/**
 * @file background.cu
 * @brief GPU per-reflection background reduction kernel.
 *
 * Reduces the per-reflection background slot tables accumulated by the Kabsch
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
 *        reflection's slot table.
 *
 * The slot table is compacted and sorted IN PLACE: occupied slots move to the
 * front of the reflection's range and are ordered ascending by value, which is
 * the precondition SparseHistogramView carries. The table is consumed once, so
 * rewriting it costs nothing and avoids per-thread scratch, keeping register
 * and local-memory use independent of NUM_BG_SLOTS.
 */
__global__ void background_reduce_kernel(BackgroundModel model,
                                         unsigned long long *d_background_slots,
                                         const uint32_t *d_background_spill,
                                         size_t num_reflections,
                                         double *d_background_mean,
                                         double *d_background_sum_value,
                                         uint32_t *d_background_count,
                                         uint8_t *d_background_success) {
    const size_t r = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= num_reflections) return;

    unsigned long long *slots = d_background_slots + r * NUM_BG_SLOTS;

    // Compact. The write index never runs ahead of the read index, so no
    // unread slot is overwritten.
    int num_entries = 0;
    for (int i = 0; i < NUM_BG_SLOTS; ++i) {
        const unsigned long long entry = slots[i];
        if (entry != 0ull) {
            slots[num_entries++] = entry;
        }
    }

    // Insertion sort ascending by value. Occupancy is a median of 2-3 entries
    // and a measured worst case in the tens, so this beats anything with a
    // better asymptotic bound.
    for (int i = 1; i < num_entries; ++i) {
        const unsigned long long entry = slots[i];
        const uint32_t value = background_entry_value(entry);
        int j = i - 1;
        while (j >= 0 && background_entry_value(slots[j]) > value) {
            slots[j + 1] = slots[j];
            --j;
        }
        slots[j + 1] = entry;
    }

    SparseHistogramView view;
    view.entries = slots;
    view.num_entries = num_entries;
    view.spill_count = d_background_spill[r];

    // Total background pixel count for this reflection (num_pixels.background),
    // spilled pixels included: they were measured, just not recorded.
    uint32_t total = view.spill_count;
    for (int i = 0; i < num_entries; ++i) {
        total += background_entry_count(slots[i]);
    }
    d_background_count[r] = total;

    BackgroundResult res;
    switch (model) {
    case BackgroundModel::Constant:
        res = tukey_constant_background(view);
        break;
    case BackgroundModel::Glm:
        res = glm_constant_background(view);
        break;
    }

    d_background_mean[r] = res.mean;
    d_background_sum_value[r] = res.weighted_sum;
    d_background_success[r] = res.valid ? 1u : 0u;
}

void compute_background(BackgroundModel model,
                        unsigned long long *d_background_slots,
                        const uint32_t *d_background_spill,
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
      d_background_slots,
      d_background_spill,
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
