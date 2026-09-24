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
 * stored entry always has a count of at least 1.
 *
 * Lock-free because a slot's KEY is immutable once claimed: only the count
 * ever changes afterwards. That makes every outcome of a racing read safe. A
 * read of zero may be stale, and the atomicCAS that follows re-checks it
 * atomically. A read of a key is either the final key or nothing at all, so a
 * key that matches is always genuinely this value's slot, and a key that does
 * not match will never later become one.
 *
 * The first probe read uses __ldcg, which bypasses the non-coherent per-SM L1
 * and reads L2, the same point global atomics serialise at. A plain load could
 * return an L1 line predating another SM's claim, and the thread would probe
 * past an occupied slot and create a second entry for the same value, breaking
 * the one-entry-per-value invariant the ordered scan depends on.
 *
 * Steady state, where the value already has a slot, is one L2 read plus one
 * 64-bit atomic add; the compare-and-swap is paid only on a value's first
 * appearance.
 *
 * Fibonacci hashing (the multiplier is 2^64/φ) is used rather than the value
 * itself so that values sharing low bits, which alias under a mask, land in
 * unrelated slots.
 *
 * @param slots This reflection's NUM_BG_SLOTS slots.
 * @param value Background pixel value to record.
 * @return false when every slot is occupied by other values, so the pixel was
 *         dropped and the caller must count it as spill.
 */
__device__ inline bool background_slot_insert(unsigned long long *slots,
                                              uint32_t value) {
    constexpr unsigned long long kSlotHashMultiplier = 0x9E3779B97F4A7C15ull;

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
        if (background_entry_value(cur) == value) {
            // The count occupies the low word and is bounded by this
            // reflection's background pixel count, so a 64-bit add of one
            // increments it without reaching the value.
            atomicAdd(&slots[idx], 1ull);
            return true;
        }
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
