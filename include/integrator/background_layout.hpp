/**
 * @file background_layout.hpp
 * @brief Host-side sizing of the per-reflection background slot tables.
 *
 * Each reflection records its background pixels in a slot table with one entry
 * per distinct value. A reflection cannot hold more distinct values than it has
 * pixels, and the bbox volume is known before any image is read, so a table
 * sized to the volume can never fill and that reflection never spills.
 *
 * Every table lives in one shared device buffer, sized to a memory budget. When
 * the tables at full size overrun the budget, the largest are trimmed to a
 * common cap, chosen as the highest value that still fits, and only a
 * reflection whose volume exceeds the cap can spill. The cap never drops below
 * a floor of 64 slots, the size of a uniform fixed-size table, so a tight
 * budget is overrun rather than spilling a reflection that the fixed-size table
 * would have held.
 *
 * The volume also counts foreground and out-of-image pixels, so it
 * overestimates the background pixel count and costs memory on large
 * shoeboxes.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// The floor: the lowest the cap may go, in slots. A tight budget lowers the
// cap, and the floor stops it at 64, so no reflection spills that a uniform
// 64-slot table would have held. At the floor the tables cost no more memory
// than that uniform table. A table for a bbox under 64 pixels stays at its
// volume whatever the cap.
constexpr uint32_t BG_SLOT_CAP_FLOOR = 64;

// Default share of free device memory, measured at allocation, that the slot
// buffer may take. The rest is headroom for the allocations made after it: one
// device image per reader thread, a reflection-index buffer per image and the
// CUDA runtime's own lazy allocations. The integrator exposes it as
// --background.memory_fraction.
constexpr double kBackgroundTableMemoryFraction = 0.25;

/**
 * @brief Where one reflection's slot table sits in the shared slot buffer.
 *
 * The table occupies slots [offset, offset + capacity). capacity == 0 marks a
 * reflection with no table: one that is never integrated, so the kernel never
 * visits it.
 */
struct BackgroundSlotRange {
    uint64_t offset = 0;
    uint32_t capacity = 0;
};

/**
 * @brief Per-reflection slot tables laid out contiguously in one buffer.
 */
struct BackgroundSlotLayout {
    std::vector<BackgroundSlotRange> ranges;  ///< One per reflection
    uint64_t total_slots = 0;                 ///< Size of the shared buffer
    std::size_t num_capped = 0;  ///< Reflections whose volume exceeds the cap
};

/**
 * @brief Table size for one reflection, in slots.
 *
 * @param volume Bbox volume in pixels, or 0 for a reflection that is not
 *        integrated.
 * @param cap Largest table, in slots.
 * @return min(volume, cap); 0 for a reflection with no table.
 */
inline uint32_t background_slot_capacity(uint64_t volume, uint32_t cap) {
    return static_cast<uint32_t>(std::min<uint64_t>(volume, cap));
}

/**
 * @brief Total slots needed by every reflection's table under a given cap.
 *
 * @param volumes Per-reflection bbox volumes in pixels, 0 when not integrated.
 * @param cap Largest table, in slots.
 */
inline uint64_t background_total_slots(const std::vector<uint64_t> &volumes,
                                       uint32_t cap) {
    // The same sum the layout's offsets reach, without building the ranges, so
    // the cap search can probe a candidate cap cheaply.
    uint64_t total = 0;
    for (uint64_t volume : volumes) {
        total += background_slot_capacity(volume, cap);
    }
    return total;
}

/**
 * @brief Largest cap whose tables fit a memory budget.
 *
 * The total never decreases as the cap grows, so a binary search over
 * [BG_SLOT_CAP_FLOOR, largest volume] finds the largest cap that fits. Past the
 * largest volume, raising the cap changes no table.
 *
 * @param volumes Per-reflection bbox volumes in pixels, 0 when not integrated.
 * @param budget_bytes Device memory available to the slot buffer.
 * @return The chosen cap, in slots. The floor, BG_SLOT_CAP_FLOOR, even when
 *         the tables at the floor overrun the budget; the caller reports the
 *         overrun.
 */
inline uint32_t choose_background_slot_cap(const std::vector<uint64_t> &volumes,
                                           std::size_t budget_bytes) {
    // Each slot is one 64-bit (value, count) word, matching the device table.
    const auto fits = [&](uint32_t cap) {
        return background_total_slots(volumes, cap) * sizeof(unsigned long long)
               <= budget_bytes;
    };

    // Upper end of the search. Past the largest volume, a larger cap changes no
    // table, so this is the cap at which no reflection is capped.
    uint64_t max_volume = 0;
    for (uint64_t volume : volumes) {
        max_volume = std::max(max_volume, volume);
    }
    // A capacity is stored in 32 bits.
    max_volume = std::min<uint64_t>(max_volume, std::numeric_limits<uint32_t>::max());

    uint32_t lo = BG_SLOT_CAP_FLOOR;  // fits, or is the floor regardless
    uint32_t hi = static_cast<uint32_t>(std::max<uint64_t>(max_volume, lo));
    // Every table at full volume fits, so no reflection can spill.
    if (fits(hi)) {
        return hi;
    }
    // Bisect between the floor and the uncapped cap.
    // Invariant: lo is returnable and hi does not fit.
    while (hi - lo > 1) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (fits(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/**
 * @brief Lay out every reflection's table in one contiguous buffer.
 *
 * Offsets are an exclusive prefix sum of the table sizes, in reflection order.
 *
 * @param volumes Per-reflection bbox volumes in pixels, 0 when not integrated.
 * @param cap Largest table, in slots.
 */
inline BackgroundSlotLayout background_slot_layout(const std::vector<uint64_t> &volumes,
                                                   uint32_t cap) {
    BackgroundSlotLayout layout;
    layout.ranges.resize(volumes.size());
    for (std::size_t r = 0; r < volumes.size(); ++r) {
        // Each table starts where the previous one ends. A reflection with no
        // table takes no slots, so it shares its offset with the next one.
        const uint32_t capacity = background_slot_capacity(volumes[r], cap);
        layout.ranges[r].offset = layout.total_slots;
        layout.ranges[r].capacity = capacity;
        layout.total_slots += capacity;

        // A capped reflection can hold more distinct values than its table, so
        // these are the only reflections that can still spill.
        if (volumes[r] > cap) {
            ++layout.num_capped;
        }
    }
    return layout;
}
