/**
 * @file test_background_layout.cc
 * @brief Host unit tests for the per-reflection slot table sizing in
 *        integrator/background_layout.hpp.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "integrator/background_layout.hpp"

// Size of one slot in the device table, a packed 64-bit (value, count) word.
// Budgets below are written in slots and converted with this.
namespace {
constexpr std::size_t kSlotBytes = sizeof(unsigned long long);
}

// One table size check: a bbox volume, the cap in force, and the slots the
// reflection should get.
struct CapacityCase {
    uint64_t volume;
    uint32_t cap;
    uint32_t expected;
};

class BackgroundSlotCapacity : public testing::TestWithParam<CapacityCase> {};

TEST_P(BackgroundSlotCapacity, SizesTheTableToTheVolumeUpToTheCap) {
    const CapacityCase c = GetParam();
    EXPECT_EQ(background_slot_capacity(c.volume, c.cap), c.expected)
      << "volume " << c.volume << " under cap " << c.cap;
}

// Covers a reflection with no table, the smallest table, a table under the
// cap, one exactly at it, one just over it, and a large bbox capped at the floor.
INSTANTIATE_TEST_SUITE_P(Volumes,
                         BackgroundSlotCapacity,
                         testing::Values(CapacityCase{0, 1000, 0},  // no table
                                         CapacityCase{1, 1000, 1},
                                         CapacityCase{300, 1000, 300},  // exact
                                         CapacityCase{1000, 1000, 1000},
                                         CapacityCase{1001, 1000, 1000},  // capped
                                         CapacityCase{500000, 64, 64}));

// Tables are packed back to back in reflection order. The zero volume is a
// reflection that is not integrated: it takes no slots, so the next table
// starts at the same offset.
TEST(BackgroundSlotLayout, OffsetsAreAnExclusivePrefixSumOfTableSizes) {
    const std::vector<uint64_t> volumes{2, 0, 9, 4};
    const BackgroundSlotLayout layout = background_slot_layout(volumes, 1000);

    ASSERT_EQ(layout.ranges.size(), volumes.size()) << "one range per reflection";
    const uint64_t expected_offsets[] = {0, 2, 2, 11};
    const uint32_t expected_capacity[] = {2, 0, 9, 4};
    for (std::size_t r = 0; r < volumes.size(); ++r) {
        EXPECT_EQ(layout.ranges[r].offset, expected_offsets[r]) << "offset of " << r;
        EXPECT_EQ(layout.ranges[r].capacity, expected_capacity[r])
          << "capacity of " << r;
    }
    EXPECT_EQ(layout.total_slots, 15u) << "buffer size is the sum of table sizes";
    EXPECT_EQ(layout.num_capped, 0u) << "no volume exceeds a cap of 1000";
}

// The capped count is what the integrator logs as the reflections that can
// still spill. A volume equal to the cap fits and is not counted.
TEST(BackgroundSlotLayout, CountsReflectionsWhoseVolumeExceedsTheCap) {
    // Cap of 64: 64 fits exactly, 65 and 1000 are capped.
    const std::vector<uint64_t> volumes{64, 65, 1000, 3};
    const BackgroundSlotLayout layout = background_slot_layout(volumes, 64);

    EXPECT_EQ(layout.num_capped, 2u) << "only volumes above 64 can still spill";
    EXPECT_EQ(layout.total_slots, 64u + 64u + 64u + 3u) << "capped tables hold 64";
}

// The cap search sizes the buffer with background_total_slots and the
// integrator allocates from background_slot_layout, so the two must agree at
// every cap, including ones either side of a volume.
TEST(BackgroundSlotLayout, TotalMatchesTheStandaloneSum) {
    const std::vector<uint64_t> volumes{1, 7, 0, 300, 4096, 33};
    for (uint32_t cap : {64u, 100u, 300u, 301u, 4095u, 5000u}) {
        EXPECT_EQ(background_slot_layout(volumes, cap).total_slots,
                  background_total_slots(volumes, cap))
          << "layout and budget search must agree at cap " << cap;
    }
}

// With an unlimited budget the search stops at the largest volume rather than
// running on to the 32-bit limit.
TEST(BackgroundSlotCap, StopsAtTheLargestVolume) {
    const std::vector<uint64_t> volumes{100, 300, 900};
    EXPECT_EQ(choose_background_slot_cap(volumes, SIZE_MAX), 900u)
      << "a cap above the largest volume changes no table";
}

// A worked case where the answer can be derived by hand. Capacities are exact,
// so the cap lands on the budget to the slot.
TEST(BackgroundSlotCap, PicksTheLargestCapThatFitsTheBudget) {
    const std::vector<uint64_t> volumes{100, 300, 900};
    // Above 300 the total is 400 + cap, so 1000 slots fit exactly at cap 600.
    EXPECT_EQ(choose_background_slot_cap(volumes, 1000 * kSlotBytes), 600u)
      << "1000 slots fit exactly at a cap of 600";
    EXPECT_EQ(choose_background_slot_cap(volumes, 999 * kSlotBytes), 599u)
      << "one slot short lowers the cap by one";
}

// A budget too small for the floor still gets the floor. The integrator
// compares the resulting total against the budget and reports the overrun.
TEST(BackgroundSlotCap, ReturnsTheFloorWhenEvenItExceedsTheBudget) {
    const std::vector<uint64_t> volumes{100, 300, 900};
    EXPECT_EQ(choose_background_slot_cap(volumes, 0), BG_SLOT_CAP_FLOOR)
      << "the caller reports the overrun; the cap never drops below the floor";
}

// Small bboxes alone never pull the cap below the floor. Their tables are
// already sized to their volumes, so the cap does not affect them.
TEST(BackgroundSlotCap, ReturnsTheFloorWhenEveryVolumeIsBelowIt) {
    const std::vector<uint64_t> volumes{5, 20, 0};
    EXPECT_EQ(choose_background_slot_cap(volumes, SIZE_MAX), BG_SLOT_CAP_FLOOR)
      << "the cap never drops below the floor";
}

// Sweeps the budget from zero to past the uncapped total. At each step the
// chosen cap must not shrink, must fit once the floor fits, and must be
// maximal: one more slot of cap would overrun. That last check is what
// confirms the binary search lands on the boundary and not just under it.
TEST(BackgroundSlotCap, IsTheLargestCapThatFitsAcrossBudgets) {
    const std::vector<uint64_t> volumes{5, 80, 700, 2500, 40, 0, 12000};
    const uint64_t floor_total = background_total_slots(volumes, BG_SLOT_CAP_FLOOR);
    uint32_t previous = BG_SLOT_CAP_FLOOR;
    for (std::size_t slots = 0; slots <= 16000; slots += 250) {
        const uint32_t cap = choose_background_slot_cap(volumes, slots * kSlotBytes);
        EXPECT_GE(cap, previous) << "a larger budget must never lower the cap";
        if (slots >= floor_total) {
            EXPECT_LE(background_total_slots(volumes, cap), slots)
              << "the chosen cap must fit a budget of " << slots;
            if (cap < 12000) {
                EXPECT_GT(background_total_slots(volumes, cap + 1), slots)
                  << "one more slot of cap must not fit a budget of " << slots;
            }
        }
        previous = cap;
    }
}

// The floor's guarantee against a fixed 64-slot table per integrated
// reflection: a table at or under the floor costs at most 64 slots, and the
// zero volume costs nothing.
TEST(BackgroundSlotCap, FloorNeverExceedsAUniformSixtyFourSlotTable) {
    const std::vector<uint64_t> volumes{1, 50, 64, 65, 100000, 0};
    const std::size_t integrated = static_cast<std::size_t>(
      std::count_if(volumes.begin(), volumes.end(), [](uint64_t v) { return v != 0; }));
    EXPECT_LE(background_total_slots(volumes, BG_SLOT_CAP_FLOOR), integrated * 64)
      << "tables at the floor must cost no more than the fixed 64-slot layout";
}
