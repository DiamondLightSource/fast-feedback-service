/**
 * @file test_background_pool.cu
 * @brief Device tests for the shared background pool in
 *        integrator/background.cuh and its reduction in integrator/background.cu.
 *
 * Every insert is launched as its own thread, so inserts race one another
 * through the same atomics the Kabsch kernel uses. Pool sizes are deliberately
 * not powers of two, since the home entry comes from a multiply-shift over any
 * capacity and the wrap at capacity is what lets a probe reach the last entry.
 * The reduction tests compare against the single-source models run on the
 * host over the histogram each reflection was fed, so they pin the sort and
 * the per-reflection runs rather than the models themselves.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include "cuda_common.hpp"
#include "integrator/background.cuh"
#include "integrator/background.hpp"
#include "integrator/background_pool.hpp"

namespace {

__global__ void insert_pixels_kernel(BackgroundPool pool,
                                     const uint32_t *reflections,
                                     const uint32_t *values,
                                     uint32_t *spill,
                                     uint32_t num_pixels) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pixels) return;
    background_accumulate(pool, reflections[i], &spill[reflections[i]], values[i]);
}

struct Pixel {
    uint32_t reflection;
    uint32_t value;
};

// A device pool and per-reflection spill counters, filled by concurrent inserts.
class DevicePool {
  public:
    DevicePool(uint32_t capacity, uint32_t num_reflections)
        : keys_(capacity), counts_(capacity), spill_(num_reflections) {
        cudaMemset(keys_.data(), 0xFF, capacity * sizeof(unsigned long long));
        cudaMemset(counts_.data(), 0, capacity * sizeof(uint32_t));
        cudaMemset(spill_.data(), 0, num_reflections * sizeof(uint32_t));
        pool_.keys = keys_.data();
        pool_.counts = counts_.data();
        pool_.capacity = capacity;
        num_reflections_ = num_reflections;
    }

    void insert(const std::vector<Pixel> &pixels) {
        std::vector<uint32_t> refl(pixels.size()), value(pixels.size());
        for (size_t i = 0; i < pixels.size(); ++i) {
            refl[i] = pixels[i].reflection;
            value[i] = pixels[i].value;
        }
        DeviceBuffer<uint32_t> d_refl(pixels.size());
        DeviceBuffer<uint32_t> d_value(pixels.size());
        d_refl.assign(refl.data());
        d_value.assign(value.data());
        const uint32_t n = static_cast<uint32_t>(pixels.size());
        insert_pixels_kernel<<<(n + 127) / 128, 128>>>(
          pool_, d_refl.data(), d_value.data(), spill_.data(), n);
        cuda_throw_error();
        cudaDeviceSynchronize();
    }

    // (reflection, value) -> count, read back from the unsorted pool.
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> contents() const {
        std::vector<unsigned long long> keys(pool_.capacity);
        std::vector<uint32_t> counts(pool_.capacity);
        keys_.extract(keys.data());
        counts_.extract(counts.data());
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> out;
        for (uint32_t i = 0; i < pool_.capacity; ++i) {
            if (keys[i] == BG_POOL_EMPTY_KEY) continue;
            out[{background_pool_key_reflection(keys[i]),
                 background_pool_key_value(keys[i])}] = counts[i];
        }
        return out;
    }

    std::vector<uint32_t> spill() const {
        std::vector<uint32_t> out(num_reflections_);
        spill_.extract(out.data());
        return out;
    }

    BackgroundPool pool() const {
        return pool_;
    }
    const uint32_t *spill_data() const {
        return spill_.data();
    }

  private:
    DeviceBuffer<unsigned long long> keys_;
    DeviceBuffer<uint32_t> counts_;
    DeviceBuffer<uint32_t> spill_;
    BackgroundPool pool_;
    uint32_t num_reflections_ = 0;
};

struct Reduced {
    std::vector<double> mean, sum;
    std::vector<uint32_t> count;
    std::vector<uint8_t> success;
    uint32_t occupied = 0;
};

Reduced reduce(BackgroundModel model, DevicePool &dp, uint32_t num_reflections) {
    DeviceBuffer<double> d_mean(num_reflections), d_sum(num_reflections);
    DeviceBuffer<uint32_t> d_count(num_reflections);
    DeviceBuffer<uint8_t> d_success(num_reflections);
    Reduced out;
    out.occupied = compute_background(model,
                                      dp.pool(),
                                      dp.spill_data(),
                                      num_reflections,
                                      d_mean.data(),
                                      d_sum.data(),
                                      d_count.data(),
                                      d_success.data(),
                                      0);
    out.mean.resize(num_reflections);
    out.sum.resize(num_reflections);
    out.count.resize(num_reflections);
    out.success.resize(num_reflections);
    d_mean.extract(out.mean.data());
    d_sum.extract(out.sum.data());
    d_count.extract(out.count.data());
    d_success.extract(out.success.data());
    return out;
}

// The model run on the host over a reflection's histogram, as packed ascending
// entries, which is what the device should reproduce exactly.
BackgroundResult host_model(BackgroundModel model,
                            const std::map<uint32_t, uint32_t> &histogram) {
    std::vector<unsigned long long> entries;
    for (const auto &[value, count] : histogram) {
        entries.push_back(background_entry_pack(value, count));
    }
    BackgroundHistogramView view{entries.data(), static_cast<int>(entries.size()), 0};
    return model == BackgroundModel::Constant ? tukey_constant_background(view)
                                              : glm_constant_background(view);
}

// Histograms for a spread of reflections: a low Poisson-like background, a
// wide one with hundreds of distinct values, one with a far outlier, and one
// with no pixels at all.
std::vector<std::map<uint32_t, uint32_t>> sample_histograms() {
    std::vector<std::map<uint32_t, uint32_t>> h(40);
    for (uint32_t r = 0; r < h.size(); ++r) {
        for (uint32_t v = 0; v < 4 + r % 9; ++v) h[r][v + r % 5] = 20 + 3 * v + r;
    }
    for (uint32_t v = 100; v < 400; ++v) h[7][v] = 1 + v % 4;  // 300 distinct
    h[11][500000] = 2;                                         // outlier
    h[23].clear();                                             // no pixels
    return h;
}

std::vector<Pixel> expand(const std::vector<std::map<uint32_t, uint32_t>> &h) {
    std::vector<Pixel> pixels;
    for (uint32_t r = 0; r < h.size(); ++r) {
        for (const auto &[value, count] : h[r]) {
            for (uint32_t c = 0; c < count; ++c) pixels.push_back({r, value});
        }
    }
    return pixels;
}

}  // namespace

// The same value in different reflections is a different key, so each
// reflection's histogram stays separate in the shared pool.
TEST(BackgroundPool, KeepsTheSameValueInDifferentReflectionsApart) {
    DevicePool dp(101, 4);
    std::vector<Pixel> pixels;
    for (uint32_t r = 0; r < 4; ++r) {
        for (int i = 0; i < 5; ++i) pixels.push_back({r, 7});
    }
    dp.insert(pixels);

    const auto contents = dp.contents();
    EXPECT_EQ(contents.size(), 4u) << "one entry per reflection for value 7";
    for (uint32_t r = 0; r < 4; ++r) {
        EXPECT_EQ(contents.at({r, 7}), 5u) << "reflection " << r << " saw 7 five times";
    }
}

// A pool no larger than the probe cap fills to its last entry without spilling.
TEST(BackgroundPool, HoldsAsManyDistinctPairsAsItHasEntries) {
    for (uint32_t capacity : {1u, 37u, kBackgroundPoolMaxProbes}) {
        DevicePool dp(capacity, 3);
        std::vector<Pixel> pixels;
        for (uint32_t i = 0; i < capacity; ++i) pixels.push_back({i % 3, 1000 + 7 * i});
        dp.insert(pixels);

        uint32_t spilled = 0;
        for (uint32_t s : dp.spill()) spilled += s;
        EXPECT_EQ(spilled, 0u) << "a full pool must not spill, capacity " << capacity;
        EXPECT_EQ(dp.contents().size(), capacity)
          << "every entry must hold one pair, capacity " << capacity;
    }
}

// Once the pool is full, each pixel of a pair that found no entry spills
// against its own reflection.
TEST(BackgroundPool, SpillsExactlyThePairsBeyondItsCapacity) {
    DevicePool dp(37, 2);
    std::vector<Pixel> pixels;
    for (uint32_t i = 0; i < 37 + 25; ++i) pixels.push_back({i % 2, 1000 + 7 * i});
    dp.insert(pixels);

    const std::vector<uint32_t> spill = dp.spill();
    EXPECT_EQ(spill[0] + spill[1], 25u) << "spill must count the excess";
    EXPECT_EQ(dp.contents().size(), 37u) << "the pool must fill before it spills";
}

// An overfull pool larger than the cap spills at least the excess and still
// counts every stored pair exactly.
TEST(BackgroundPool, GivesUpAfterTheProbeCapWhenOverfull) {
    const uint32_t capacity = 20 * kBackgroundPoolMaxProbes;
    const uint32_t pairs = capacity + 500;
    DevicePool dp(capacity, 4);
    std::vector<Pixel> pixels;
    for (uint32_t i = 0; i < pairs; ++i) {
        for (int c = 0; c < 3; ++c) pixels.push_back({i % 4, 1000 + 7 * i});
    }
    dp.insert(pixels);

    uint32_t spilled = 0, stored = 0;
    for (uint32_t s : dp.spill()) spilled += s;
    const auto contents = dp.contents();
    for (const auto &[pair, count] : contents) {
        stored += count;
        EXPECT_EQ(count, 3u) << "each stored pair saw all three of its pixels";
    }
    EXPECT_EQ(stored + spilled, pixels.size()) << "every pixel is stored or spilled";
    EXPECT_GE(spilled, 3u * 500) << "at least the excess pairs spill";
    EXPECT_LE(contents.size(), capacity) << "no more pairs than entries";
}

// Repeated values share one entry, so a small pool holds many pixels when the
// distinct pairs fit.
TEST(BackgroundPool, CountsRepeatedValuesInOneEntryEach) {
    DevicePool dp(10, 1);
    std::vector<Pixel> pixels;
    for (uint32_t i = 0; i < 3000; ++i) pixels.push_back({0, 5 + i % 10});
    dp.insert(pixels);

    EXPECT_EQ(dp.spill()[0], 0u) << "10 distinct values fit 10 entries";
    for (const auto &[pair, count] : dp.contents()) {
        EXPECT_EQ(count, 300u) << "value " << pair.second << " is seen 300 times";
    }
}

// The reduction hands each reflection its own run, ascending by value, so the
// device result equals the host model over the histogram that reflection was
// fed, for both models. A run handed to the wrong reflection or out of order
// would move the quartiles by whole counts, far outside the GLM's tolerance.
TEST(BackgroundPoolReduction, MatchesTheHostModelForEveryReflection) {
    const auto histograms = sample_histograms();
    const uint32_t n = static_cast<uint32_t>(histograms.size());
    uint32_t distinct = 0;
    for (const auto &h : histograms) distinct += static_cast<uint32_t>(h.size());

    for (BackgroundModel model : {BackgroundModel::Constant, BackgroundModel::Glm}) {
        DevicePool dp(n * 64, n);
        dp.insert(expand(histograms));
        const Reduced out = reduce(model, dp, n);
        const char *model_name = model == BackgroundModel::Glm ? "GLM" : "Tukey";

        EXPECT_EQ(out.occupied, distinct) << "occupied entries are the distinct pairs";
        for (uint32_t r = 0; r < n; ++r) {
            const BackgroundResult expected = host_model(model, histograms[r]);
            uint32_t pixels = 0;
            for (const auto &[value, count] : histograms[r]) pixels += count;

            EXPECT_EQ(out.count[r], pixels) << "pixel count of reflection " << r;
            EXPECT_EQ(out.success[r], expected.valid ? 1u : 0u)
              << "validity of reflection " << r;
            if (expected.valid) {
                // nvcc contracts a*b+c into one fused multiply-add and the host
                // compiler does not, so the iterative GLM agrees to rounding
                // rather than bit for bit. The Tukey mean is one division of
                // exact integer sums and does match exactly.
                const double tol = model == BackgroundModel::Glm ? 1e-12 : 0.0;
                EXPECT_NEAR(out.mean[r], expected.mean, tol * std::abs(expected.mean))
                  << model_name << " mean of reflection " << r;
                EXPECT_NEAR(out.sum[r],
                            expected.weighted_sum,
                            tol * std::abs(expected.weighted_sum))
                  << model_name << " weighted sum of reflection " << r;
            }
        }
    }
}

// A reflection that lost pixels to a full pool is rejected, and its pixel
// count still includes the spilled ones.
TEST(BackgroundPoolReduction, RejectsEveryReflectionThatSpilled) {
    const auto histograms = sample_histograms();
    const uint32_t n = static_cast<uint32_t>(histograms.size());
    DevicePool dp(150, n);  // fewer entries than distinct pairs
    dp.insert(expand(histograms));
    const std::vector<uint32_t> spill = dp.spill();
    const Reduced out = reduce(BackgroundModel::Constant, dp, n);

    uint32_t spilled_reflections = 0;
    for (uint32_t r = 0; r < n; ++r) {
        uint32_t pixels = 0;
        for (const auto &[value, count] : histograms[r]) pixels += count;
        EXPECT_EQ(out.count[r], pixels)
          << "spilled pixels still count, reflection " << r;
        if (spill[r] > 0) {
            ++spilled_reflections;
            EXPECT_EQ(out.success[r], 0u) << "reflection " << r << " spilled";
        }
    }
    EXPECT_GT(spilled_reflections, 0u) << "the pool is too small for every pair";
    EXPECT_EQ(out.occupied, 150u) << "a pool that spilled is full";
}
