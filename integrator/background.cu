/**
 * @file background.cu
 * @brief GPU per-reflection background reduction.
 *
 * Reduces the shared background pool accumulated by the Kabsch kernel into
 * per-reflection background estimates, using the single-source model code in
 * integrator/background.hpp so the result matches the baseline CPU path.
 *
 * A radix sort of the pool keys puts each reflection's entries next to each
 * other, ascending by value, because a key carries the reflection index above
 * the value. Each reflection then finds its run by binary search and reads it
 * as an ordinary sorted histogram. The pool is sized to a few dozen entries per
 * reflection, so the sort covers a few million keys and runs once per sweep.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <stdexcept>

#include "cuda_common.hpp"
#include "integrator.cuh"
#include "integrator/background.cuh"
#include "integrator/background.hpp"
#include "integrator/background_pool.hpp"

namespace {
constexpr int BACKGROUND_REDUCE_THREADS = 128;

/// First index in sorted keys[0, n) whose key is not less than target.
__device__ uint32_t lower_bound_key(const unsigned long long *keys,
                                    uint32_t n,
                                    unsigned long long target) {
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (keys[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}
}  // namespace

/**
 * @brief One thread per reflection: evaluate the background model over that
 *        reflection's run of the sorted pool.
 *
 * The run is repacked into (value, count) entries in the key buffer the sort
 * did not finish in, so BackgroundHistogramView sees the same layout the
 * baseline builds. The runs do not overlap, so each thread
 * writes only its own range.
 */
__global__ void background_reduce_kernel(BackgroundModel model,
                                         const unsigned long long *d_sorted_keys,
                                         const uint32_t *d_sorted_counts,
                                         unsigned long long *d_entries,
                                         uint32_t capacity,
                                         const uint32_t *d_background_spill,
                                         size_t num_reflections,
                                         double *d_background_mean,
                                         double *d_background_sum_value,
                                         uint32_t *d_background_count,
                                         uint8_t *d_background_success,
                                         uint32_t *d_occupied) {
    const size_t r = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= num_reflections) return;

    // Empty keys sort after every real key, so the first one marks the end of
    // the occupied entries.
    if (r == 0) {
        *d_occupied = lower_bound_key(d_sorted_keys, capacity, BG_POOL_EMPTY_KEY);
    }

    const uint32_t reflection = static_cast<uint32_t>(r);
    const uint32_t begin =
      lower_bound_key(d_sorted_keys, capacity, background_pool_key(reflection, 0));
    const uint32_t end =
      lower_bound_key(d_sorted_keys, capacity, background_pool_key(reflection + 1, 0));

    // Repack the run, already ascending by value, and total its pixels.
    // Spilled pixels count towards num_pixels.background: they were measured,
    // just not recorded.
    uint32_t total = d_background_spill[r];
    for (uint32_t i = begin; i < end; ++i) {
        d_entries[i] = background_entry_pack(
          background_pool_key_value(d_sorted_keys[i]), d_sorted_counts[i]);
        total += d_sorted_counts[i];
    }
    d_background_count[r] = total;

    BackgroundHistogramView view;
    view.entries = d_entries + begin;
    view.num_entries = static_cast<int>(end - begin);
    view.spill_count = d_background_spill[r];

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

uint32_t compute_background(BackgroundModel model,
                            BackgroundPool pool,
                            const uint32_t *d_background_spill,
                            size_t num_reflections,
                            double *d_background_mean,
                            double *d_background_sum_value,
                            uint32_t *d_background_count,
                            uint8_t *d_background_success,
                            cudaStream_t stream) {
    if (num_reflections == 0 || pool.capacity == 0) return 0;

    // Only the bits a real key can set take part in the sort: the 32 value bits
    // and enough above them for the largest reflection index. An empty key has
    // all of those bits set, so it still sorts last.
    int end_bit = 32;
    while (end_bit < 64 && (num_reflections >> (end_bit - 32)) != 0) {
        ++end_bit;
    }

    // A 32-bit count keeps CUB on 32-bit offsets, the same width the pool
    // indexes its entries with.
    const uint32_t n = pool.capacity;

    // DoubleBuffer keeps the peak at two copies of the pool; the other form
    // allocates a third. The key buffer the sort did not finish in holds the
    // repacked entries.
    DeviceBuffer<unsigned long long> d_spare_keys(pool.capacity);
    DeviceBuffer<uint32_t> d_spare_counts(pool.capacity);
    cub::DoubleBuffer<unsigned long long> keys(pool.keys, d_spare_keys.data());
    cub::DoubleBuffer<uint32_t> counts(pool.counts, d_spare_counts.data());
    size_t temp_bytes = 0;
    cudaError_t err = cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes, keys, counts, n, 0, end_bit, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(fmt::format("Background pool sort sizing failed: {}",
                                             cudaGetErrorString(err)));
    }
    DeviceBuffer<uint8_t> d_temp(temp_bytes);
    err = cub::DeviceRadixSort::SortPairs(
      d_temp.data(), temp_bytes, keys, counts, n, 0, end_bit, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
          fmt::format("Background pool sort failed: {}", cudaGetErrorString(err)));
    }

    DeviceBuffer<uint32_t> d_occupied(1);
    const unsigned int blocks = static_cast<unsigned int>(
      (num_reflections + BACKGROUND_REDUCE_THREADS - 1) / BACKGROUND_REDUCE_THREADS);
    background_reduce_kernel<<<blocks, BACKGROUND_REDUCE_THREADS, 0, stream>>>(
      model,
      keys.Current(),
      counts.Current(),
      keys.Alternate(),
      pool.capacity,
      d_background_spill,
      num_reflections,
      d_background_mean,
      d_background_sum_value,
      d_background_count,
      d_background_success,
      d_occupied.data());

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(fmt::format("Background reduction launch failed: {}",
                                             cudaGetErrorString(err)));
    }

    // The scratch buffers are freed on return, so the stream has to finish
    // with them first.
    cudaStreamSynchronize(stream);
    uint32_t occupied = 0;
    d_occupied.extract(&occupied);
    return occupied;
}
