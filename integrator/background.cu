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
constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_WARP_MASK = 0xffffffffu;
}  // namespace

/**
 * @brief One warp per reflection: compact that reflection's slot table, then
 *        evaluate the background model over it on lane 0.
 *
 * The slot table is compacted and sorted IN PLACE: occupied slots move to the
 * front of the reflection's range and are ordered ascending by value, which is
 * the precondition BackgroundHistogramView carries. The table is consumed once, so
 * rewriting it costs nothing and avoids per-thread scratch, keeping register
 * and local-memory use independent of the table size.
 *
 * Tables are sized to the bbox volume, so they run to thousands of slots and
 * are almost all empty. The compaction reads every slot, and a warp reads 32
 * consecutive slots in one coalesced 256-byte pass where a lone thread would
 * issue 32 separate loads. The sort and the model touch only the occupied
 * entries, a median of 2-3, so they stay serial on lane 0.
 */
__global__ void background_reduce_kernel(BackgroundModel model,
                                         unsigned long long *d_background_slots,
                                         const BackgroundSlotRange *d_background_ranges,
                                         const uint32_t *d_background_spill,
                                         size_t num_reflections,
                                         double *d_background_mean,
                                         double *d_background_sum_value,
                                         uint32_t *d_background_count,
                                         uint8_t *d_background_success) {
    const size_t r =
      (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / WARP_SIZE;
    const unsigned lane = threadIdx.x % WARP_SIZE;
    // r is warp-uniform, so a whole warp leaves together and the ballots below
    // always see a full warp.
    if (r >= num_reflections) return;

    const BackgroundSlotRange range = d_background_ranges[r];
    unsigned long long *slots = d_background_slots + range.offset;

    // Compact, 32 slots per pass. The ballot marks the occupied lanes, and each
    // occupied lane writes to the running count plus the occupied lanes below
    // it. A write lands at or before the slot its lane read, and a pass reads
    // every slot before any lane writes, so no unread slot is overwritten.
    uint32_t compacted = 0;
    for (uint32_t base = 0; base < range.capacity; base += WARP_SIZE) {
        const uint32_t i = base + lane;
        const unsigned long long entry = (i < range.capacity) ? slots[i] : 0ull;
        const unsigned occupied = __ballot_sync(FULL_WARP_MASK, entry != 0ull);
        if (entry != 0ull) {
            slots[compacted + __popc(occupied & ((1u << lane) - 1u))] = entry;
        }
        compacted += __popc(occupied);
    }
    // Makes the other lanes' compacted writes visible to lane 0.
    __syncwarp();
    if (lane != 0) return;
    const int num_entries = static_cast<int>(compacted);

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

    BackgroundHistogramView view;
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
                        const BackgroundSlotRange *d_background_ranges,
                        const uint32_t *d_background_spill,
                        size_t num_reflections,
                        double *d_background_mean,
                        double *d_background_sum_value,
                        uint32_t *d_background_count,
                        uint8_t *d_background_success,
                        cudaStream_t stream) {
    if (num_reflections == 0) return;

    // One warp per reflection.
    const size_t threads = num_reflections * WARP_SIZE;
    const unsigned int blocks = static_cast<unsigned int>(
      (threads + BACKGROUND_REDUCE_THREADS - 1) / BACKGROUND_REDUCE_THREADS);

    background_reduce_kernel<<<blocks, BACKGROUND_REDUCE_THREADS, 0, stream>>>(
      model,
      d_background_slots,
      d_background_ranges,
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
