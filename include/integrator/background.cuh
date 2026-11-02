/**
 * @file background.cuh
 * @brief GPU background accumulation into the shared pool, and its reduction.
 *
 * The Kabsch kernel records every background pixel into the shared pool
 * (integrator/background_pool.hpp) through background_accumulate(). After the
 * image loop, compute_background() reduces each reflection's histogram in the
 * pool into a background estimate (mean, inlier weighted sum, pixel count,
 * success) using the selected BackgroundModel. The reduction runs entirely on
 * the device and reuses the single-source model code in
 * integrator/background.hpp, so it produces the same result as the baseline
 * CPU path.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "integrator/background.hpp"
#include "integrator/background_pool.hpp"

// The accumulation helpers below are device code: they use atomics and the
// cache-control load intrinsic. The header is also included from host-only
// translation units for compute_background()'s declaration, so they are
// visible only under nvcc.
#if defined(__CUDACC__)

/**
 * @brief Insert one background pixel value into the shared pool.
 *
 * Open addressing with linear probing over the whole pool, keyed on
 * (reflection, value). An entry's key is immutable once claimed, so a stale
 * read can report an entry as empty but never as holding the wrong key. The
 * atomicCAS re-checks that case at L2, and is therefore what enforces one
 * entry per key. __ldcg keeps the probe read out of the per-SM L1, which is
 * not coherent between SMs, so a claimed entry does not read as empty and
 * waste a compare-and-swap.
 *
 * @param pool The shared pool.
 * @param reflection Index of the reflection the pixel belongs to.
 * @param value Background pixel value to record, at most INT32_MAX.
 * @return false when no free entry was found within kBackgroundPoolMaxProbes,
 *         so the caller must count the pixel as spill.
 */
__device__ inline bool background_pool_insert(const BackgroundPool &pool,
                                              uint32_t reflection,
                                              uint32_t value) {
    const unsigned long long key = background_pool_key(reflection, value);

    // 2^64/φ rounded odd, giving Fibonacci hashing (Knuth, TAOCP vol. 3, 6.4).
    // The high 32 bits of the product step 0.618 of a turn per unit of value,
    // so one reflection's consecutive values spread rather than cluster. The
    // reflection index, in the high word of the key, adds index times the
    // constant's low word, which is odd, so the same value in different
    // reflections hashes to different homes.
    constexpr unsigned long long kPoolHashMultiplier = 0x9E3779B97F4A7C15ull;
    const uint32_t hash = static_cast<uint32_t>((key * kPoolHashMultiplier) >> 32);

    // Home entry floor(capacity * hash / 2^32), the multiply-shift range reduction
    uint32_t idx = __umulhi(hash, pool.capacity);

    // Keys never move, so a key stored within the cap is found within it again.
    const uint32_t max_probes = min(pool.capacity, kBackgroundPoolMaxProbes);
    for (uint32_t probe = 0; probe < max_probes; ++probe) {
        unsigned long long cur = __ldcg(&pool.keys[idx]);
        if (cur == BG_POOL_EMPTY_KEY) {
            cur = atomicCAS(&pool.keys[idx], BG_POOL_EMPTY_KEY, key);
            // On success cur is still empty, the entry is now this key's, and
            // the count below is its first pixel. Otherwise cur holds whichever
            // key won the race.
            if (cur == BG_POOL_EMPTY_KEY) cur = key;
        }
        if (cur == key) {
            atomicAdd(&pool.counts[idx], 1u);
            return true;
        }
        // A step of one that wraps at capacity, so a pool no larger than the
        // cap is searched in full.
        if (++idx == pool.capacity) {
            idx = 0;
        }
    }
    return false;
}

/**
 * @brief Record one background pixel value for a reflection.
 *
 * A pixel that finds no free entry is counted as spill; the reduction fails
 * any reflection with spill, since the lost values are of unknown magnitude.
 *
 * @param pool The shared pool.
 * @param reflection Index of the reflection the pixel belongs to.
 * @param spill The reflection's spill counter.
 * @param value Background pixel value to record.
 */
__device__ inline void background_accumulate(const BackgroundPool &pool,
                                             uint32_t reflection,
                                             uint32_t *spill,
                                             uint32_t value) {
    if (!background_pool_insert(pool, reflection, value)) {
        atomicAdd(spill, 1u);
    }
}

#endif  // __CUDACC__

/**
 * @brief Reduce the background pool into per-reflection background estimates.
 *
 * Sorts the pool by key, which groups each reflection's entries and orders
 * them by value, then evaluates the model over each reflection's run with one
 * thread per reflection. The pool is consumed and must not be read afterwards.
 *
 * @param model Background model to apply (Constant = Tukey; Glm = robust-Poisson GLM)
 * @param pool Device pool filled by the Kabsch kernel
 * @param d_background_spill Device per-reflection count of pixels dropped by a full pool
 * @param num_reflections Number of reflections
 * @param d_background_mean Output: per-reflection background level (per pixel)
 * @param d_background_sum_value Output: per-reflection inlier weighted sum (background.sum.value)
 * @param d_background_count Output: per-reflection total background pixel count
 * @param d_background_success Output: per-reflection success flag (1 = estimate valid)
 * @param stream CUDA stream for async execution
 * @return Number of occupied pool entries. Synchronises the stream.
 */
uint32_t compute_background(BackgroundModel model,
                            BackgroundPool pool,
                            const uint32_t *d_background_spill,
                            size_t num_reflections,
                            double *d_background_mean,
                            double *d_background_sum_value,
                            uint32_t *d_background_count,
                            uint8_t *d_background_success,
                            cudaStream_t stream);
