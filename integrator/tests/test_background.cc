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

#include <cstdint>
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
}

// Overflow-tail pixels clip to the Huber bound regardless of their exact value,
// so the overflow count alone reproduces the DIALS result. N = 89 (4 overflow).
TEST(GlmConstantBackground, OverflowTailClips) {
    std::vector<uint32_t> bins(NUM_BG_BINS, 0);
    bins[2] = 10;
    bins[3] = 20;
    bins[4] = 30;
    bins[5] = 25;

    BackgroundResult r = glm_constant_background(view_of(bins, 4));
    ASSERT_TRUE(r.valid);
    EXPECT_NEAR(r.mean, 4.0257619071, kDialsParityTol);
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
