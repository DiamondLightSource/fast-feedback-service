/**
 * @file background_pool.hpp
 * @brief Shared device pool holding every reflection's background histogram.
 *
 * Each background pixel is recorded as one entry per distinct (reflection,
 * value) pair, in a single open-addressed hash table shared by every
 * reflection. A reflection needs one entry per distinct background value, and
 * sharing one pool sized to the average lets reflections with many values use
 * the space others leave unused, so no reflection has a cap of its own.
 *
 * The pool is sized to a fixed number of entries per reflection, so its cost is
 * known before any image is read. A pixel that finds no free entry within the
 * probe cap is counted as spill against its reflection, and the integrator logs
 * how full the pool got.
 *
 * A key packs the reflection index into the high 32 bits and the pixel value
 * into the low 32, so sorting the keys groups each reflection's entries
 * together and orders them by value within it, which is the order the
 * background models require.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "integrator/background.hpp"

// Default pool size, in entries per reflection on average. A high-background
// 3600-image sweep measured 12.7 distinct values per reflection, so 64 leaves
// a margin of about 5x. The integrator exposes it as
// --background.pool_entries_per_reflection.
constexpr uint32_t kBackgroundPoolEntriesPerReflection = 64;

// Most entries an insert inspects before counting its pixel as spill, so a full
// pool fails fast instead of scanning every entry.
constexpr uint32_t kBackgroundPoolMaxProbes = 128;

// Key of an empty entry. Pixel values above INT32_MAX are dropped before they
// reach the pool, so the low word of a real key is never all ones and no real
// key equals this. It also sorts after every real key.
constexpr unsigned long long BG_POOL_EMPTY_KEY = ~0ull;

/**
 * @brief Device view of the background pool.
 *
 * keys and counts are parallel arrays of capacity entries. An entry is empty
 * while its key is BG_POOL_EMPTY_KEY; its count is the number of pixels with
 * that reflection and value.
 */
struct BackgroundPool {
    unsigned long long *keys = nullptr;
    uint32_t *counts = nullptr;
    uint32_t capacity = 0;
};

/// @brief Pool key for one reflection's pixel value.
FFS_HD inline unsigned long long background_pool_key(uint32_t reflection,
                                                     uint32_t value) {
    return (static_cast<unsigned long long>(reflection) << 32)
           | static_cast<unsigned long long>(value);
}

/// @brief Reflection index of a pool key.
FFS_HD inline uint32_t background_pool_key_reflection(unsigned long long key) {
    return static_cast<uint32_t>(key >> 32);
}

/// @brief Pixel value of a pool key.
FFS_HD inline uint32_t background_pool_key_value(unsigned long long key) {
    return static_cast<uint32_t>(key & 0xFFFFFFFFull);
}
