/**
 * @file background.cc
 * @brief Host adapter from the baseline BackgroundAggregator histogram to the
 *        shared, device-safe constant background model.
 */

#include "integrator/background.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

// Upper bound on the contiguous histogram from the aggregator.
// Background pixel values above this are folded into the overflow tail; for a
// constant background these are extreme outliers that Tukey rejects, so the
// estimate is unaffected and large hot-pixel-sized values need no allocation.
namespace {
constexpr int kMaxBins = 4096;
}

// Tukey outlier-rejecting constant background. Delegates to the single-source
// tukey_constant_background() so the baseline and the GPU run identical math;
// this function only flattens the aggregator's array and overflow map into a
// contiguous ConstHistogramView.
std::tuple<double, double> compute_background_constant_3d(
  const BackgroundAggregator &data) {
    if (data.num_pixels() == 0) {
        throw std::runtime_error("No background pixels available");
    }

    const auto &small_hist = data.small_hist();
    const auto *large_hist = data.large_hist();  // may be nullptr

    // Determine how many contiguous integer bins to allocate: enough to
    // cover the small array and any in-range map keys, capped at kMaxBins.
    int num_bins = static_cast<int>(small_hist.size());
    if (large_hist != nullptr) {
        for (const auto &[value, count] : *large_hist) {
            if (value >= num_bins && value < kMaxBins) {
                num_bins = value + 1;
            }
        }
    }

    std::vector<uint32_t> bins(static_cast<size_t>(num_bins), 0);
    uint32_t overflow_count = 0;

    for (std::size_t v = 0; v < small_hist.size(); ++v) {
        bins[v] = static_cast<uint32_t>(small_hist[v]);
    }
    if (large_hist != nullptr) {
        for (const auto &[value, count] : *large_hist) {
            if (value >= 0 && value < num_bins) {
                bins[static_cast<size_t>(value)] += static_cast<uint32_t>(count);
            } else {
                overflow_count += static_cast<uint32_t>(count);
            }
        }
    }

    ConstHistogramView view{bins.data(), num_bins, overflow_count};
    BackgroundResult result = tukey_constant_background(view);
    if (!result.valid) {
        throw std::runtime_error(
          "No background data remaining after outlier rejection");
    }

    return {result.mean, result.weighted_sum};
}
