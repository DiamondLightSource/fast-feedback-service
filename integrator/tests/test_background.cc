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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "integrator/background.hpp"

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
// same iteration. Because the shared core and DIALS run the same
// algorithm with the same stopping rule, their results agree far more
// tightly than 1e-3 (~1e-11 in practice). 1e-6 sits between that real
// agreement and 1e-3: loose enough to absorb the documented H = N*b vs
// H += b divergence and FP error.
namespace {
constexpr double kDialsParityTol = 1e-6;
}  // namespace

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

// Half the pixels sit far above any plausible background. The shared core
// holds their values exactly, like the unbounded dials-like baseline, so the
// two agree instead of the shared core rejecting the reflection.
TEST(ConstantBackgroundImplComparison, AgreeOnValuesFarAboveTheBackground) {
    BackgroundAggregator agg;
    for (int v = 0; v <= 9; ++v) agg.add(v);  // 10 low pixels
    add_n(agg, 5000, 10);                     // 10 far above the rest

    BackgroundResult shared = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant);
    BackgroundResult dials =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::DialsIndependent);

    ASSERT_TRUE(dials.valid) << "the dials-like baseline is unbounded";
    ASSERT_TRUE(shared.valid) << "the shared core holds any value exactly";
    EXPECT_DOUBLE_EQ(shared.mean, dials.mean) << "both span the full range";
    EXPECT_DOUBLE_EQ(shared.weighted_sum, dials.weighted_sum) << "identical inliers";
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

// The cases below drive the model functions directly over hand-built
// histograms, which bypasses the baseline host adapter.

namespace {

// Build the sorted entry list a slot table would produce for a bins vector
// indexed by value, which is a compact way to write a fixture.
std::vector<unsigned long long> entries_of(const std::vector<uint32_t> &bins) {
    std::vector<unsigned long long> entries;
    for (std::size_t v = 0; v < bins.size(); ++v) {
        if (bins[v] != 0) {
            entries.push_back(background_entry_pack(static_cast<uint32_t>(v), bins[v]));
        }
    }
    return entries;
}

// Build an entry list from explicit (value, count) pairs. Sorted here so
// callers need not be.
std::vector<unsigned long long> entries_from(
  std::vector<std::pair<uint32_t, uint32_t>> value_counts) {
    std::sort(value_counts.begin(), value_counts.end());
    std::vector<unsigned long long> entries;
    entries.reserve(value_counts.size());
    for (const auto &[value, count] : value_counts) {
        entries.push_back(background_entry_pack(value, count));
    }
    return entries;
}

// Build an aggregator holding the given (value, count) pairs.
BackgroundAggregator aggregator_of(
  const std::vector<std::pair<int, int>> &value_counts) {
    BackgroundAggregator agg;
    for (const auto &[value, count] : value_counts) {
        for (int n = 0; n < count; ++n) agg.add(value);
    }
    return agg;
}

// The adapter must reproduce a directly built view exactly, not approximately,
// so these compare with EXPECT_DOUBLE_EQ rather than a tolerance.
void expect_same_result(const BackgroundResult &a, const BackgroundResult &b) {
    EXPECT_EQ(a.valid, b.valid) << "validity must match";
    if (a.valid && b.valid) {
        EXPECT_DOUBLE_EQ(a.mean, b.mean) << "background level must match";
        EXPECT_DOUBLE_EQ(a.weighted_sum, b.weighted_sum) << "inlier sum must match";
    }
}

SparseHistogramView sparse_view_of(const std::vector<unsigned long long> &entries,
                                   uint32_t spill = 0) {
    return SparseHistogramView{entries.data(), static_cast<int>(entries.size()), spill};
}

}  // namespace

TEST(TukeyConstantBackground, EmptyHistogramFails) {
    std::vector<unsigned long long> entries;
    BackgroundResult r = tukey_constant_background(sparse_view_of(entries));
    EXPECT_FALSE(r.valid) << "an empty histogram must not produce an estimate";
}

// Uniform spread 0..9 (one pixel each). No outliers: mean is the plain mean.
TEST(TukeyConstantBackground, UniformNoOutliers) {
    std::vector<uint32_t> bins(64, 0);
    for (int v = 0; v <= 9; ++v) bins[v] = 1;  // N = 10
    auto entries = entries_of(bins);

    BackgroundResult r = tukey_constant_background(sparse_view_of(entries));
    ASSERT_TRUE(r.valid) << "a clean uniform spread must produce an estimate";
    EXPECT_DOUBLE_EQ(r.weighted_sum, 45.0) << "inlier sum over 0..9";
    EXPECT_DOUBLE_EQ(r.mean, 4.5) << "mean of 0..9";
}

TEST(TukeyConstantBackground, HighOutlierRejected) {
    std::vector<uint32_t> bins(64, 0);
    for (int v = 0; v <= 9; ++v) bins[v] = 1;
    bins[60] = 1;  // clear outlier well above q3 + 1.5*IQR
    auto entries = entries_of(bins);

    BackgroundResult r = tukey_constant_background(sparse_view_of(entries));
    ASSERT_TRUE(r.valid) << "one outlier must not fail the estimate";
    EXPECT_DOUBLE_EQ(r.weighted_sum, 45.0) << "the outlier must not enter the sum";
    EXPECT_DOUBLE_EQ(r.mean, 4.5) << "the outlier must not shift the mean";
}

TEST(TukeyConstantBackground, ConstantValue) {
    auto entries = entries_from({{5, 20}});

    BackgroundResult r = tukey_constant_background(sparse_view_of(entries));
    ASSERT_TRUE(r.valid) << "a zero-IQR histogram is still estimable";
    EXPECT_DOUBLE_EQ(r.mean, 5.0) << "mean of a single repeated value";
    EXPECT_DOUBLE_EQ(r.weighted_sum, 100.0) << "20 pixels of value 5";
}

// A spread wide enough that the upper fence q3 + 1.5*IQR runs well past the
// widest value present. Entries carry their own values, so there is no range
// to run out of and the spread is estimated normally.
TEST(TukeyConstantBackground, WideSpreadAccepted) {
    std::vector<uint32_t> bins(16, 1);  // N = 16, uniform 0..15
    auto entries = entries_of(bins);

    BackgroundResult r = tukey_constant_background(sparse_view_of(entries));
    ASSERT_TRUE(r.valid) << "there is no range limit to trip";
    // q1=3, q3=11, IQR=8 -> bounds [-9, 23]; all of 0..15 survive.
    EXPECT_DOUBLE_EQ(r.weighted_sum, 120.0) << "inlier sum over 0..15";
    EXPECT_DOUBLE_EQ(r.mean, 7.5) << "mean of 0..15";
}

// Values far above any plausible background are held exactly, with no tail.
TEST(TukeyConstantBackground, LargeValuesRepresentedExactly) {
    std::vector<std::pair<uint32_t, uint32_t>> value_counts;
    for (uint32_t v = 5000; v <= 5009; ++v) value_counts.push_back({v, 1});
    auto entries = entries_from(value_counts);

    BackgroundResult r = tukey_constant_background(sparse_view_of(entries));
    ASSERT_TRUE(r.valid) << "a background of thousands of counts must be estimable";
    EXPECT_DOUBLE_EQ(r.mean, 5004.5) << "mean of 5000..5009";
    EXPECT_DOUBLE_EQ(r.weighted_sum, 50045.0) << "inlier sum over 5000..5009";
}

// A full slot table loses pixels of unknown value, so the estimate is refused
// rather than computed from a truncated histogram.
TEST(TukeyConstantBackground, SpillRejected) {
    std::vector<uint32_t> bins(64, 0);
    for (int v = 0; v <= 9; ++v) bins[v] = 1;
    auto entries = entries_of(bins);

    BackgroundResult r = tukey_constant_background(sparse_view_of(entries, 1));
    EXPECT_FALSE(r.valid) << "any spill must fail the reflection";
}

// The DIALS parity fixtures. The reference means and their regeneration recipe
// are documented above.
TEST(GlmConstantBackground, TightLowNoOutliers) {
    auto entries = entries_from({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}});

    BackgroundResult r = glm_constant_background(sparse_view_of(entries));
    ASSERT_TRUE(r.valid) << "the fit must converge on a clean low background";
    EXPECT_NEAR(r.mean, 4.0304431542, kDialsParityTol) << "DIALS RobustPoissonMean";
    EXPECT_DOUBLE_EQ(r.weighted_sum, r.mean * 24.0) << "GLM sum is mean over all N";
}

TEST(GlmConstantBackground, HighOutlierDownweighted) {
    auto entries = entries_from({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}, {120, 1}});

    BackgroundResult r = glm_constant_background(sparse_view_of(entries));
    ASSERT_TRUE(r.valid) << "one outlier must not fail the fit";
    EXPECT_NEAR(r.mean, 4.1427022177, kDialsParityTol) << "DIALS RobustPoissonMean";
    EXPECT_DOUBLE_EQ(r.weighted_sum, r.mean * 25.0) << "the outlier counts in N";
}

// A high tail is recorded at its true value, and reproduces the DIALS number
// for a folded-in tail, since psi clips it either way.
TEST(GlmConstantBackground, HighTailRecordedExactly) {
    auto entries = entries_from({{2, 10}, {3, 20}, {4, 30}, {5, 25}, {5000, 4}});

    BackgroundResult r = glm_constant_background(sparse_view_of(entries));
    ASSERT_TRUE(r.valid) << "a recorded high tail must still fit";
    EXPECT_NEAR(r.mean, 4.0257619071, kDialsParityTol) << "DIALS RobustPoissonMean";
    EXPECT_DOUBLE_EQ(r.weighted_sum, r.mean * 89.0) << "N counts the tail pixels";
}

TEST(GlmConstantBackground, ModerateLevel) {
    auto entries = entries_from({{48, 4}, {50, 10}, {52, 8}, {55, 3}, {60, 2}});

    BackgroundResult r = glm_constant_background(sparse_view_of(entries));
    ASSERT_TRUE(r.valid) << "a higher background must still fit";
    EXPECT_NEAR(r.mean, 51.6834964586, kDialsParityTol) << "DIALS RobustPoissonMean";
    EXPECT_DOUBLE_EQ(r.weighted_sum, r.mean * 27.0) << "GLM sum is mean over all N";
}

TEST(GlmConstantBackground, TooFewPixelsFails) {
    auto entries = entries_from({{3, 1}, {4, 1}, {5, 1}, {6, 1}, {7, 1}});  // N = 5

    BackgroundResult r = glm_constant_background(sparse_view_of(entries));
    EXPECT_FALSE(r.valid) << "fewer than kGlmMinPixels must not be fitted";
}

TEST(GlmConstantBackground, SpillRejected) {
    auto entries = entries_from({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}});

    BackgroundResult r = glm_constant_background(sparse_view_of(entries, 1));
    EXPECT_FALSE(r.valid) << "any spill must fail the reflection";
}

// The cases below test the host adapter,
// compute_background_constant_3d(..., SharedCore, ...), which builds the entry
// list from a BackgroundAggregator the way the device builds it from a slot
// table.

TEST(BackgroundAdapter, EmptyAggregatorFails) {
    BackgroundAggregator agg;
    BackgroundResult r = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant);
    EXPECT_FALSE(r.valid) << "an aggregator with no pixels must not be estimated";
}

TEST(BackgroundAdapter, MatchesDirectViewTukey) {
    BackgroundAggregator agg = aggregator_of({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}});
    auto entries = entries_from({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}});

    expect_same_result(
      compute_background_constant_3d(
        agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant),
      tukey_constant_background(sparse_view_of(entries)));
}

TEST(BackgroundAdapter, MatchesDirectViewGlm) {
    BackgroundAggregator agg = aggregator_of({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}});
    auto entries = entries_from({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}});

    expect_same_result(compute_background_constant_3d(
                         agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Glm),
                       glm_constant_background(sparse_view_of(entries)));
}

// Values spanning the aggregator's small array and its large map, so the
// adapter's ordering of the two halves is exercised against a directly built
// entry list.
TEST(BackgroundAdapter, SpansSmallArrayAndLargeMap) {
    BackgroundAggregator agg =
      aggregator_of({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}, {70, 1}, {120, 1}});
    auto entries =
      entries_from({{2, 3}, {3, 5}, {4, 8}, {5, 6}, {6, 2}, {70, 1}, {120, 1}});

    expect_same_result(
      compute_background_constant_3d(
        agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant),
      tukey_constant_background(sparse_view_of(entries)));
}

// A background of thousands of counts is held exactly, with no tail and no
// rejection.
TEST(BackgroundAdapter, LargeValueEstimatedExactly) {
    BackgroundAggregator agg = aggregator_of({{5000, 10}, {5001, 10}});

    BackgroundResult r = compute_background_constant_3d(
      agg, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant);
    ASSERT_TRUE(r.valid) << "the adapter must estimate a high background";
    EXPECT_DOUBLE_EQ(r.mean, 5000.5) << "mean of ten 5000s and ten 5001s";
    EXPECT_DOUBLE_EQ(r.weighted_sum, 100010.0) << "inlier sum over both values";
}

// Negative sentinels are dropped by the aggregator and must not reach the
// entry list, matching the GPU kernel.
TEST(BackgroundAdapter, NegativeSentinelDropped) {
    BackgroundAggregator with_sentinels = aggregator_of({{4, 10}, {5, 10}});
    add_n(with_sentinels, -1, 5);
    BackgroundAggregator clean = aggregator_of({{4, 10}, {5, 10}});

    expect_same_result(
      compute_background_constant_3d(
        with_sentinels, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant),
      compute_background_constant_3d(
        clean, ConstantBackgroundImpl::SharedCore, BackgroundModel::Constant));
}
