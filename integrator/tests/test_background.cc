/**
 * @file test_background.cc
 * @brief Host unit tests for the single-source constant (Tukey/IQR) background
 *        model in integrator/background.hpp.
 *
 * These exercise tukey_constant_background() directly over hand-built
 * histograms with known quartiles and outliers, independent of CUDA. The same
 * function is compiled for the device, so locking its behaviour here also pins
 * the GPU reduction's result.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
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

namespace {

// Add a value to a BackgroundAggregator a given number of times.
void add_n(BackgroundAggregator &agg, int value, int times) {
    for (int i = 0; i < times; ++i) agg.add(value);
}

}  // namespace

// With only low, outlier-free values, the independent dials-like baseline and
// the shared core agree exactly.
TEST(ConstantBackgroundImplComparison, AgreeOnCleanLowValues) {
    BackgroundAggregator agg;
    for (int v = 0; v <= 9; ++v) agg.add(v);  // N = 10, one pixel each

    auto [dials_mean, dials_sum] =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::DialsIndependent);
    auto [shared_mean, shared_sum] =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::SharedCore);

    EXPECT_DOUBLE_EQ(dials_mean, 4.5);
    EXPECT_DOUBLE_EQ(dials_sum, 45.0);
    EXPECT_DOUBLE_EQ(shared_mean, dials_mean);
    EXPECT_DOUBLE_EQ(shared_sum, dials_sum);
}

// Negative sentinel pixels are counted by the dials-like baseline (and pulled
// into the inlier mean here) but dropped by the shared core, so the two
// diverge. This is exactly the true-to-dials behaviour the baseline keeps.
TEST(ConstantBackgroundImplComparison, NegativesCountedByDialsDroppedByShared) {
    BackgroundAggregator agg;
    for (int v = 0; v <= 9; ++v) add_n(agg, v, 10);  // 100 low pixels
    add_n(agg, -1, 4);                               // 4 negative sentinels

    auto [dials_mean, dials_sum] =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::DialsIndependent);
    auto [shared_mean, shared_sum] =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::SharedCore);

    // Dials counts the four -1 pixels as inliers: sum 450 - 4 over 104 pixels.
    EXPECT_DOUBLE_EQ(dials_sum, 446.0);
    EXPECT_DOUBLE_EQ(dials_mean, 446.0 / 104.0);
    // Shared drops the sentinels: sum 450 over 100 pixels.
    EXPECT_DOUBLE_EQ(shared_sum, 450.0);
    EXPECT_DOUBLE_EQ(shared_mean, 4.5);
}

// A large fraction of pixels above NUM_BG_BINS overflows the shared core's
// bounded histogram, which it rejects; the unbounded dials-like baseline still
// produces an estimate from the full range.
TEST(ConstantBackgroundImplComparison, HighOverflowRejectedOnlyByShared) {
    BackgroundAggregator agg;
    for (int v = 0; v <= 9; ++v) agg.add(v);  // 10 low pixels
    add_n(agg, 5000, 10);                     // 10 pixels above NUM_BG_BINS

    // Shared core: overflow fraction (50%) exceeds the permitted limit.
    EXPECT_THROW(
      compute_background_constant_3d(agg, ConstantBackgroundImpl::SharedCore),
      std::runtime_error);

    // Dials-like baseline: unbounded, so it still returns an estimate.
    auto [dials_mean, dials_sum] =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::DialsIndependent);
    EXPECT_GT(dials_sum, 0.0);
    EXPECT_TRUE(std::isfinite(dials_mean));
}

// The default implementation is the independent dials-like baseline.
TEST(ConstantBackgroundImplComparison, DefaultIsDialsIndependent) {
    BackgroundAggregator agg;
    for (int v = 0; v <= 9; ++v) add_n(agg, v, 10);
    add_n(agg, -1, 4);

    auto [default_mean, default_sum] = compute_background_constant_3d(agg);
    auto [dials_mean, dials_sum] =
      compute_background_constant_3d(agg, ConstantBackgroundImpl::DialsIndependent);

    EXPECT_DOUBLE_EQ(default_mean, dials_mean);
    EXPECT_DOUBLE_EQ(default_sum, dials_sum);
}
