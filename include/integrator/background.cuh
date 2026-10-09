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

// The accumulation helpers below are device code: they use atomics and the
// cache-control load intrinsic. The header is also included from host-only
// translation units for compute_background()'s declaration, so they are
// visible only under nvcc.
#if defined(__CUDACC__)

/**
 * @brief Insert one background pixel value into a reflection's slot table.
 *
 * Open addressing with linear probing over NUM_BG_SLOTS slots. A slot is one
 * 64-bit word packing (value, count); an all-zero word is empty, since a
 * stored entry always has a count of at least 1. The home slot comes from
 * Fibonacci hashing (see kSlotHashMultiplier).
 *
 * A slot's key is immutable once claimed, so a stale read can report a slot as
 * empty but never as holding the wrong key. The atomicCAS re-checks that case
 * at L2, and is therefore what enforces one entry per value. __ldcg keeps the
 * probe read out of the per-SM L1, which is not coherent between SMs, so a
 * claimed slot does not read as empty and waste a compare-and-swap.
 *
 * @param slots This reflection's NUM_BG_SLOTS slots.
 * @param value Background pixel value to record.
 * @return false when every slot is occupied by other values, so the pixel was
 *         dropped and the caller must count it as spill.
 */
__device__ inline bool background_slot_insert(unsigned long long *slots,
                                              uint32_t value) {
    // 2^64/φ rounded odd, giving Fibonacci hashing (Knuth, TAOCP vol. 3, 6.4).
    // Taking the high bits of the product is floor(NUM_BG_SLOTS*frac(value/φ)),
    // a step of 0.618 of a turn around the table per unit of value, so
    // consecutive values spread rather than cluster. A mask steps one
    // slot instead, aliasing values NUM_BG_SLOTS apart onto the same
    // home slot.
    //
    // Collisions are resolved by probing, not prevented, and placement
    // does not reach the result: the reduction sorts by value. A run of
    // consecutive values stays collision-free to about half the table,
    // against a measured occupancy of 3 values and a worst case of 18.
    // The cost is locality, since a mask would keep low values on one
    // cache line.
    constexpr unsigned long long kSlotHashMultiplier = 0x9E3779B97F4A7C15ull;

    // Home slot. The shift leaves exactly background_slot_bits(), so no mask.
    uint32_t idx = static_cast<uint32_t>(
      (static_cast<unsigned long long>(value) * kSlotHashMultiplier)
      >> (64 - background_slot_bits()));

    for (int probe = 0; probe < NUM_BG_SLOTS; ++probe) {
        unsigned long long cur = __ldcg(&slots[idx]);
        if (cur == 0ull) {
            const unsigned long long old =
              atomicCAS(&slots[idx], 0ull, background_entry_pack(value, 1));
            if (old == 0ull) {
                return true;  // claimed the slot, count starts at 1
            }
            cur = old;  // lost the race; old holds whichever value won
        }
        // The stored key, not the hash, settles identity.
        if (background_entry_value(cur) == value) {
            // The count is bounded by this reflection's background pixel count,
            // so a 64-bit add of one cannot carry into the value.
            atomicAdd(&slots[idx], 1ull);
            return true;
        }
        // A step of one over a power-of-two table visits every slot, so the
        // loop bound is reached only once the table is full.
        idx = (idx + 1) & (NUM_BG_SLOTS - 1);
    }
    return false;
}

/**
 * @brief Record one background pixel value for a reflection.
 *
 * A pixel that finds no free slot is counted as spill; the reduction fails any
 * reflection with spill, since the lost values are of unknown magnitude.
 */
__device__ inline void background_accumulate(unsigned long long *d_background_slots,
                                             uint32_t *d_background_spill,
                                             size_t refl_idx,
                                             uint32_t value) {
    if (!background_slot_insert(&d_background_slots[refl_idx * NUM_BG_SLOTS], value)) {
        atomicAdd(&d_background_spill[refl_idx], 1u);
    }
}

#endif  // __CUDACC__

/**
 * @brief Reduce per-reflection background slot tables into background estimates.
 *
 * One thread handles one reflection. The slot table is rewritten in place (compacted and sorted), so
 * it must not be read again afterwards.
 *
 * @param model Background model to apply (Constant = Tukey; Glm = robust-Poisson GLM)
 * @param d_background_slots Device slot tables, num_reflections * NUM_BG_SLOTS entries
 * @param d_background_spill Device per-reflection count of pixels dropped by a full table
 * @param num_reflections Number of reflections
 * @param d_background_mean Output: per-reflection background level (per pixel)
 * @param d_background_sum_value Output: per-reflection inlier weighted sum (background.sum.value)
 * @param d_background_count Output: per-reflection total background pixel count
 * @param d_background_success Output: per-reflection success flag (1 = estimate valid)
 * @param stream CUDA stream for async execution
 */
void compute_background(BackgroundModel model,
                        unsigned long long *d_background_slots,
                        const uint32_t *d_background_spill,
                        size_t num_reflections,
                        double *d_background_mean,
                        double *d_background_sum_value,
                        uint32_t *d_background_count,
                        uint8_t *d_background_success,
                        cudaStream_t stream);
