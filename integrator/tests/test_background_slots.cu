/**
 * @file test_background_slots.cu
 * @brief Device tests for the per-reflection background slot table in
 *        integrator/background.cuh.
 *
 * Each test launches one thread per pixel value against a single table, so
 * every insert races the others through the same atomics the Kabsch kernel
 * uses. The table sizes are deliberately not powers of two: the home slot
 * comes from a multiply-shift over any capacity, and the wrap at capacity is
 * what lets a probe reach the last slot.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "cuda_common.hpp"
#include "integrator/background.cuh"

namespace {

__global__ void insert_values_kernel(unsigned long long *slots,
                                     uint32_t capacity,
                                     uint32_t *spill,
                                     const uint32_t *values,
                                     uint32_t num_values) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_values) return;
    background_accumulate(slots, capacity, spill, values[i]);
}

struct TableContents {
    std::vector<unsigned long long> slots;
    uint32_t spill = 0;
};

// Insert every value into one table of the given capacity, concurrently.
TableContents insert_values(uint32_t capacity, const std::vector<uint32_t> &values) {
    // A capacity of 0 still gets one slot of storage, so the buffer is valid to
    // pass; the insert never touches it.
    DeviceBuffer<unsigned long long> d_slots(std::max<uint32_t>(capacity, 1));
    DeviceBuffer<uint32_t> d_spill(1);
    DeviceBuffer<uint32_t> d_values(values.size());
    cudaMemset(
      d_slots.data(), 0, std::max<uint32_t>(capacity, 1) * sizeof(unsigned long long));
    cudaMemset(d_spill.data(), 0, sizeof(uint32_t));
    d_values.assign(values.data());

    const uint32_t n = static_cast<uint32_t>(values.size());
    insert_values_kernel<<<(n + 127) / 128, 128>>>(
      d_slots.data(), capacity, d_spill.data(), d_values.data(), n);
    cuda_throw_error();
    cudaDeviceSynchronize();

    TableContents out;
    out.slots.resize(std::max<uint32_t>(capacity, 1));
    d_slots.extract(out.slots.data());
    d_spill.extract(&out.spill);
    out.slots.resize(capacity);
    return out;
}

std::vector<uint32_t> distinct_values(uint32_t n) {
    std::vector<uint32_t> values(n);
    for (uint32_t i = 0; i < n; ++i) values[i] = 1000 + 7 * i;
    return values;
}

uint32_t occupied(const TableContents &t) {
    uint32_t n = 0;
    for (unsigned long long s : t.slots) n += (s != 0ull);
    return n;
}

}  // namespace

// The bbox-volume guarantee: a table with one slot per value holds every
// distinct value, filling the last slot, with nothing spilled.
TEST(BackgroundSlotTable, HoldsAsManyDistinctValuesAsItHasSlots) {
    for (uint32_t capacity : {1u, 37u, 100u, 1000u}) {
        const TableContents t = insert_values(capacity, distinct_values(capacity));
        EXPECT_EQ(t.spill, 0u) << "a full table must not spill, capacity " << capacity;
        EXPECT_EQ(occupied(t), capacity)
          << "every slot must hold one value, capacity " << capacity;
        for (unsigned long long s : t.slots) {
            EXPECT_EQ(background_entry_count(s), 1u)
              << "each distinct value is seen once, capacity " << capacity;
        }
    }
}

// A capped table that sees more distinct values than it has slots spills
// exactly the excess, which is what the reduction uses to reject it.
TEST(BackgroundSlotTable, SpillsExactlyTheValuesBeyondItsCapacity) {
    for (uint32_t capacity : {1u, 37u, 100u}) {
        const TableContents t = insert_values(capacity, distinct_values(capacity + 25));
        EXPECT_EQ(t.spill, 25u) << "spill must count the excess, capacity " << capacity;
        EXPECT_EQ(occupied(t), capacity)
          << "the table must fill before it spills, capacity " << capacity;
    }
}

// Repeated values share one slot, so a table smaller than the pixel count
// holds them all when the distinct count fits.
TEST(BackgroundSlotTable, CountsRepeatedValuesInOneSlotEach) {
    std::vector<uint32_t> values;
    for (uint32_t i = 0; i < 3000; ++i) values.push_back(5 + i % 10);
    const TableContents t = insert_values(10, values);

    EXPECT_EQ(t.spill, 0u) << "10 distinct values fit 10 slots";
    for (unsigned long long s : t.slots) {
        EXPECT_EQ(background_entry_count(s), 300u)
          << "value " << background_entry_value(s) << " is seen 300 times";
    }
}

// A reflection that is not integrated has capacity 0. Any pixel that reaches
// it spills rather than writing outside the buffer.
TEST(BackgroundSlotTable, SpillsEveryValueWhenThereIsNoTable) {
    const TableContents t = insert_values(0, distinct_values(5));
    EXPECT_EQ(t.spill, 5u) << "a table of capacity 0 stores nothing";
}
