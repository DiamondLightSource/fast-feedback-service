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
