/**
 * @file test_background.cc
 * @brief Host unit tests for the single-source background models in
 *        integrator/background.hpp.
 *
 * These exercise tukey_constant_background() and glm_constant_background()
 * directly over hand-built histograms, independent of CUDA. The same functions
 * are compiled for the device, so locking their behaviour here also pins the
 * GPU reduction's result. The GLM expected values come from DIALS
 * RobustPoissonMean run on the expanded histograms, so the tests assert parity
 * with DIALS.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "integrator/background.hpp"

namespace {

// Build a ConstHistogramView over a caller-owned bins vector.
ConstHistogramView view_of(const std::vector<uint32_t> &bins, uint32_t overflow = 0) {
    return ConstHistogramView{bins.data(), static_cast<int>(bins.size()), overflow};
}

}  // namespace

// Empty histogram -> no estimate.
TEST(TukeyConstantBackground, EmptyHistogramFails) {
    std::vector<uint32_t> bins(16, 0);
    BackgroundResult r = tukey_constant_background(view_of(bins));
    EXPECT_FALSE(r.valid);
}

// Uniform spread 0..9 (one pixel each). No outliers: mean is the plain mean.
TEST(TukeyConstantBackground, UniformNoOutliers) {
    std::vector<uint32_t> bins(64, 0);
    for (int v = 0; v <= 9; ++v) bins[v] = 1;  // N = 10

    BackgroundResult r = tukey_constant_background(view_of(bins));
    ASSERT_TRUE(r.valid);
    // q1=2, q3=6, IQR=4 -> bounds [-4, 12]; all of 0..9 survive.
    EXPECT_DOUBLE_EQ(r.weighted_sum, 45.0);
    EXPECT_DOUBLE_EQ(r.mean, 4.5);
}

// A single extreme value in the overflow tail must be rejected and must not
// perturb the mean of the inliers.
TEST(TukeyConstantBackground, HighOutlierInOverflowRejected) {
    std::vector<uint32_t> bins(64, 0);
    for (int v = 0; v <= 9; ++v) bins[v] = 1;
    const uint32_t overflow = 1;  // one pixel with value >= num_bins (e.g. 5000)

    BackgroundResult r = tukey_constant_background(view_of(bins, overflow));
    ASSERT_TRUE(r.valid);
    // Inliers remain 0..9; the overflow pixel is above the upper bound.
    EXPECT_DOUBLE_EQ(r.weighted_sum, 45.0);
    EXPECT_DOUBLE_EQ(r.mean, 4.5);
}

// A high outlier inside the binned range (not overflow) is also rejected.
TEST(TukeyConstantBackground, HighOutlierInBinsRejected) {
    std::vector<uint32_t> bins(64, 0);
    for (int v = 0; v <= 9; ++v) bins[v] = 1;
    bins[60] = 1;  // clear outlier well above q3 + 1.5*IQR

    BackgroundResult r = tukey_constant_background(view_of(bins));
    ASSERT_TRUE(r.valid);
    EXPECT_DOUBLE_EQ(r.weighted_sum, 45.0);
    EXPECT_DOUBLE_EQ(r.mean, 4.5);
}

// A spread wide enough that the upper fence q3 + 1.5*IQR reaches num_bins is
// rejected, even with an empty overflow tail: the range is too small to apply
// Tukey rejection, so the estimate is untrustworthy.
TEST(TukeyConstantBackground, UpperFenceReachingOverflowRejected) {
    std::vector<uint32_t> bins(16, 1);  // N = 16, uniform 0..15, no overflow

    BackgroundResult r = tukey_constant_background(view_of(bins));
    // q1=3, q3=11, IQR=8 -> upper_bound = 23 >= num_bins (16).
    EXPECT_FALSE(r.valid);
}

// Degenerate: every pixel has the same value -> IQR 0, mean equals that value.
TEST(TukeyConstantBackground, ConstantValue) {
    std::vector<uint32_t> bins(64, 0);
    bins[5] = 20;  // N = 20, all value 5

    BackgroundResult r = tukey_constant_background(view_of(bins));
    ASSERT_TRUE(r.valid);
    EXPECT_DOUBLE_EQ(r.mean, 5.0);
    EXPECT_DOUBLE_EQ(r.weighted_sum, 100.0);
}

// Reference means below were produced by DIALS RobustPoissonMean (tuning
// constant 1.345, tolerance 1e-3, max_iter 100) on the expanded histograms.
// Matching them confirms the single-source GLM core reproduces DIALS.
//
// To regenerate (using DIALS): expand each histogram back into a flat
// list of pixel values, since RobustPoissonMean takes the raw values,
// not bin counts. For TightLowNoOutliers the list is [2,2,2, 3x5, 4x8,
// 5x6, 6,6] (24 values). The constructor is RobustPoissonMean(Y, mean0,
// c=1.345, tolerance=1e-3, max_iter=100), where mean0 is the median
// seed (the sorted element at index N/2, matching the seed in
// glm_constant_background) and overflow pixels become any value past
// the bin range (the Huber clip makes their exact value irrelevant).
// Run:
//   from dials.algorithms.background.glm import RobustPoissonMean
//   from scitbx.array_family import flex
//   m = RobustPoissonMean(flex.double(values), 4.0, 1.345, 1e-3, 100)
//   print(m.mean())
// and paste the result. The values are frozen constants, so any change
// to the tuning constant, tolerance, or max_iter above invalidates them
// and they must be regenerated this way.

// This parity tolerance is distinct from the GLM's own convergence
// tolerance (kGlmTolerance = 1e-3, the DIALS default at which the IRLS
// loop stops). That 1e-3 is matched to DIALS so both fits halt at the
// same iteration. Because our core and DIALS run the same algorithm
// with the same stopping rule, their results agree far more tightly
// than 1e-3 (~1e-11 in practice). 1e-6 sits between that real agreement
// and 1e-3: loose enough to absorb the documented H = N*b vs H += b
// divergence and FP error
namespace {
constexpr double kDialsParityTol = 1e-6;
}  // namespace

// Tight low background, no outliers. N = 24, median seed 4.
TEST(GlmConstantBackground, TightLowNoOutliers) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    bins[2] = 3;
    bins[3] = 5;
    bins[4] = 8;
    bins[5] = 6;
    bins[6] = 2;

    BackgroundResult r = glm_constant_background(view_of(bins));
    ASSERT_TRUE(r.valid);
    EXPECT_NEAR(r.mean, 4.0304431542, kDialsParityTol);
    // GLM models every background pixel at mean, so the reported sum is mean*N.
    EXPECT_DOUBLE_EQ(r.weighted_sum, r.mean * 24.0);
}

// A single in-range high outlier is down-weighted, not rejected outright, so it
// shifts the mean slightly. N = 25, median seed 4.
TEST(GlmConstantBackground, HighOutlierDownweighted) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    bins[2] = 3;
    bins[3] = 5;
    bins[4] = 8;
    bins[5] = 6;
    bins[6] = 2;
    bins[120] = 1;

    BackgroundResult r = glm_constant_background(view_of(bins));
    ASSERT_TRUE(r.valid);
    EXPECT_NEAR(r.mean, 4.1427022177, kDialsParityTol);
    // Sum is mean over all N background pixels, the outlier included.
    EXPECT_DOUBLE_EQ(r.weighted_sum, r.mean * 25.0);
}

// Overflow-tail pixels clip to the Huber bound regardless of their
// exact value, so the overflow count alone reproduces the DIALS result.
// N = 89 (4 overflow).
TEST(GlmConstantBackground, OverflowTailClips) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    bins[2] = 10;
    bins[3] = 20;
    bins[4] = 30;
    bins[5] = 25;

    BackgroundResult r = glm_constant_background(view_of(bins, 4));
    ASSERT_TRUE(r.valid);
    EXPECT_NEAR(r.mean, 4.0257619071, kDialsParityTol);
    // N counts the 4 overflow pixels too, so the sum is mean over all 89.
    EXPECT_DOUBLE_EQ(r.weighted_sum, r.mean * 89.0);
}

// Higher background level. N = 27, median seed 50.
TEST(GlmConstantBackground, ModerateLevel) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    bins[48] = 4;
    bins[50] = 10;
    bins[52] = 8;
    bins[55] = 3;
    bins[60] = 2;

    BackgroundResult r = glm_constant_background(view_of(bins));
    ASSERT_TRUE(r.valid);
    EXPECT_NEAR(r.mean, 51.6834964586, kDialsParityTol);
    EXPECT_DOUBLE_EQ(r.weighted_sum, r.mean * 27.0);
}

// Fewer than kGlmMinPixels background pixels -> no estimate (matches DIALS,
// which asserts num_background >= min_pixels).
TEST(GlmConstantBackground, TooFewPixelsFails) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    for (int v = 3; v < 8; ++v) bins[v] = 1;  // N = 5 < kGlmMinPixels

    BackgroundResult r = glm_constant_background(view_of(bins));
    EXPECT_FALSE(r.valid);
}

// Too much of the background in the overflow tail -> range too small, rejected.
TEST(GlmConstantBackground, ExcessiveOverflowRejected) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    bins[3] = 10;
    bins[4] = 10;  // 20 in range

    BackgroundResult r = glm_constant_background(view_of(bins, 20));  // 50% overflow
    EXPECT_FALSE(r.valid);
}

// The tests above feed a ConstHistogramView straight into the model
// functions, which bypasses the baseline host adapter. The cases below
// tests compute_background_constant_3d, which flattens a
// BackgroundAggregator into the same view before dispatching. They pin
// the adapter's binning (the small/large split versus the NUM_BG_BINS
// range, negative-sentinel drop, overflow tail) by asserting it
// reproduces the already-DIALS-pinned direct-view results.
namespace {

// Build an aggregator by adding each (value, count) pair count times, matching
// how the Kabsch kernel feeds pixels in one at a time.
BackgroundAggregator aggregator_of(
  const std::vector<std::pair<int, int>> &value_counts) {
    BackgroundAggregator agg;
    for (const auto &[value, count] : value_counts) {
        for (int n = 0; n < count; ++n) agg.add(value);
    }
    return agg;
}

// Assert two BackgroundResults are bit-for-bit identical (same bins in, so the
// adapter and the direct view must agree exactly, not just to a tolerance).
void expect_same_result(const BackgroundResult &a, const BackgroundResult &b) {
    EXPECT_EQ(a.valid, b.valid);
    if (a.valid && b.valid) {
        EXPECT_DOUBLE_EQ(a.mean, b.mean);
        EXPECT_DOUBLE_EQ(a.weighted_sum, b.weighted_sum);
    }
}

}  // namespace

// No background pixels -> no estimate, for either model.
TEST(BackgroundAdapter, EmptyAggregatorFails) {
    BackgroundAggregator agg;  // num_pixels() == 0
    EXPECT_FALSE(compute_background_constant_3d(
                   agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant)
                   .valid);
    EXPECT_FALSE(compute_background_constant_3d(
                   agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Glm)
                   .valid);
}

// The adapter's flattened histogram must reproduce the direct-view GLM result.
// Same data as GlmConstantBackground.TightLowNoOutliers.
TEST(BackgroundAdapter, MatchesDirectViewGlm) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    bins[2] = 3;
    bins[3] = 5;
    bins[4] = 8;
    bins[5] = 6;
    bins[6] = 2;
    BackgroundResult direct = glm_constant_background(view_of(bins));

    BackgroundAggregator agg = aggregator_of({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}});
    BackgroundResult adapted = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Glm);

    ASSERT_TRUE(adapted.valid);
    expect_same_result(adapted, direct);
}

// Same parity check for the Tukey/constant model.
// Same data as TukeyConstantBackground.UniformNoOutliers.
TEST(BackgroundAdapter, MatchesDirectViewTukey) {
    std::vector<uint32_t> bins(64, 0);
    for (int v = 0; v <= 9; ++v) bins[v] = 1;
    BackgroundResult direct = tukey_constant_background(view_of(bins));

    std::vector<std::pair<int, int>> value_counts;
    for (int v = 0; v <= 9; ++v) value_counts.push_back({v, 1});
    BackgroundAggregator agg = aggregator_of(value_counts);
    BackgroundResult adapted = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant);

    ASSERT_TRUE(adapted.valid);
    expect_same_result(adapted, direct);
}

// A value in [VECTOR_LIMIT, NUM_BG_BINS) lands in the aggregator's large_hist
// map but must be flattened to an in-range bin, not the overflow tail. 120 is
// the outlier value from GlmConstantBackground.HighOutlierDownweighted.
TEST(BackgroundAdapter, LargeHistValueBinnedInRange) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    bins[2] = 3;
    bins[3] = 5;
    bins[4] = 8;
    bins[5] = 6;
    bins[6] = 2;
    bins[120] = 1;  // in-range, not overflow
    BackgroundResult direct = glm_constant_background(view_of(bins));

    BackgroundAggregator agg =
      aggregator_of({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}, {120, 1}});
    BackgroundResult adapted = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Glm);

    ASSERT_TRUE(adapted.valid);
    expect_same_result(adapted, direct);
}

// Values at or above NUM_BG_BINS must be counted in the overflow tail, where
// they clip to the Huber bound. Same data as GlmConstantBackground.OverflowTailClips
// (4 overflow pixels), here fed as concrete out-of-range values.
TEST(BackgroundAdapter, OverflowTailCounted) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    bins[2] = 10;
    bins[3] = 20;
    bins[4] = 30;
    bins[5] = 25;
    BackgroundResult direct = glm_constant_background(view_of(bins, 4));

    BackgroundAggregator agg =
      aggregator_of({{2, 10}, {3, 20}, {4, 30}, {5, 25}, {5000, 4}});
    BackgroundResult adapted = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Glm);

    ASSERT_TRUE(adapted.valid);
    expect_same_result(adapted, direct);
}

// Negative pixels are sentinels, dropped by the adapter rather than counted.
// Adding them must not change the estimate.
TEST(BackgroundAdapter, NegativeSentinelDropped) {
    std::vector<std::pair<int, int>> base = {{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}};
    BackgroundAggregator clean = aggregator_of(base);
    BackgroundResult without = compute_background_constant_3d(
      clean, ConstantBackgroundImpl::SharedCore, BackgroundModel::Glm);

    std::vector<std::pair<int, int>> with_sentinels = base;
    with_sentinels.push_back({-1, 3});
    with_sentinels.push_back({-100, 2});
    BackgroundAggregator dirty = aggregator_of(with_sentinels);
    BackgroundResult with = compute_background_constant_3d(
      dirty, ConstantBackgroundImpl::SharedCore, BackgroundModel::Glm);

    ASSERT_TRUE(without.valid);
    expect_same_result(with, without);
}

// Too much of the background in the overflow tail trips the adapter's
// overflow-fraction guard before the model runs.
TEST(BackgroundAdapter, ExcessiveOverflowRejected) {
    // 20 in range, 20 overflow -> 50% overflow, above kBackgroundMaxOverflowFraction.
    BackgroundAggregator agg = aggregator_of({{3, 10}, {4, 10}, {5000, 20}});
    EXPECT_FALSE(compute_background_constant_3d(
                   agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Glm)
                   .valid);
    EXPECT_FALSE(compute_background_constant_3d(
                   agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant)
                   .valid);
}

namespace {

// Add a value to a BackgroundAggregator a given number of times.
void add_n(BackgroundAggregator &agg, int value, int times) {
    for (int i = 0; i < times; ++i) agg.add(value);
}

}  // namespace

// With only low, outlier-free values, the independent dials-like baseline and
// the shared core agree exactly. Both run the Tukey/IQR (Constant) model.
TEST(ConstantBackgroundImplComparison, AgreeOnCleanLowValues) {
    BackgroundAggregator agg;
    for (int v = 0; v <= 9; ++v) agg.add(v);  // N = 10, one pixel each

    BackgroundResult dials =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::DialsIndependent);
    BackgroundResult shared = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant);

    ASSERT_TRUE(dials.valid);
    ASSERT_TRUE(shared.valid);
    EXPECT_DOUBLE_EQ(dials.mean, 4.5);
    EXPECT_DOUBLE_EQ(dials.weighted_sum, 45.0);
    EXPECT_DOUBLE_EQ(shared.mean, dials.mean);
    EXPECT_DOUBLE_EQ(shared.weighted_sum, dials.weighted_sum);
}

// Negative values are garbage pixels that slipped past the mask, not real
// background measurements. The aggregator drops them at the source, so both the
// dials-like baseline and the shared core see only the 100 clean pixels and
// agree on the estimate.
TEST(ConstantBackgroundImplComparison, NegativesDroppedBeforeEstimation) {
    BackgroundAggregator agg;
    for (int v = 0; v <= 9; ++v) add_n(agg, v, 10);  // 100 low pixels
    add_n(agg, -1, 4);                               // 4 negatives, dropped

    EXPECT_EQ(agg.num_pixels(), 100);

    BackgroundResult dials =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::DialsIndependent);
    BackgroundResult shared = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant);

    ASSERT_TRUE(dials.valid);
    ASSERT_TRUE(shared.valid);
    // Both see sum 450 over the 100 retained pixels.
    EXPECT_DOUBLE_EQ(dials.weighted_sum, 450.0);
    EXPECT_DOUBLE_EQ(dials.mean, 4.5);
    EXPECT_DOUBLE_EQ(shared.weighted_sum, dials.weighted_sum);
    EXPECT_DOUBLE_EQ(shared.mean, dials.mean);
}

// A large fraction of pixels above NUM_BG_BINS overflows the shared core's
// bounded histogram, which it rejects (valid=false); the unbounded dials-like
// baseline still produces an estimate from the full range.
TEST(ConstantBackgroundImplComparison, HighOverflowRejectedOnlyByShared) {
    BackgroundAggregator agg;
    for (int v = 0; v <= 9; ++v) agg.add(v);  // 10 low pixels
    add_n(agg, 5000, 10);                     // 10 pixels above NUM_BG_BINS

    // Shared core: overflow fraction (50%) exceeds the permitted limit.
    BackgroundResult shared = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant);
    EXPECT_FALSE(shared.valid);

    // Dials-like baseline: unbounded, so it still returns an estimate.
    BackgroundResult dials =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::DialsIndependent);
    ASSERT_TRUE(dials.valid);
    EXPECT_GT(dials.weighted_sum, 0.0);
    EXPECT_TRUE(std::isfinite(dials.mean));
}

// The default implementation is the independent dials-like baseline.
TEST(ConstantBackgroundImplComparison, DefaultIsDialsIndependent) {
    BackgroundAggregator agg;
    for (int v = 0; v <= 9; ++v) add_n(agg, v, 10);
    add_n(agg, -1, 4);

    BackgroundResult def = compute_background_constant_3d(agg);
    BackgroundResult dials =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::DialsIndependent);

    ASSERT_TRUE(def.valid);
    ASSERT_TRUE(dials.valid);
    EXPECT_DOUBLE_EQ(def.mean, dials.mean);
    EXPECT_DOUBLE_EQ(def.weighted_sum, dials.weighted_sum);
}
