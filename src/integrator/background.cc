/**
 * @file background.cc
 * @brief Host adapter from the baseline BackgroundAggregator histogram to the
 *        shared, device-safe constant background model.
 */

#include "integrator/background.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

// Tukey outlier-rejecting constant background. Delegates to the single-source
// tukey_constant_background() so the baseline and the GPU run identical math;
// this function only flattens the aggregator's array and overflow map into a
// contiguous ConstHistogramView over the shared NUM_BG_BINS range.
std::tuple<double, double> compute_background_constant_3d(
  const BackgroundAggregator &data) {
    if (data.num_pixels() == 0) {
        throw std::runtime_error("No background pixels available");
    }

    const auto &small_hist = data.small_hist();
    const auto *large_hist = data.large_hist();  // may be nullptr

    // Bin into the shared [0, NUM_BG_BINS) range exactly as the GPU kernel does,
    // so the host baseline and the device reduction see identical histograms:
    // negative sentinel values are dropped, values at or above NUM_BG_BINS go to
    // the overflow tail.
    std::vector<uint32_t> bins(static_cast<size_t>(NUM_BG_BINS), 0);
    uint32_t overflow_count = 0;

    for (std::size_t v = 0; v < small_hist.size() && v < NUM_BG_BINS; ++v) {
        bins[v] = static_cast<uint32_t>(small_hist[v]);
    }
    if (large_hist != nullptr) {
        for (const auto &[value, count] : *large_hist) {
            // A negative value is a sentinel/garbage pixel, not a real
            // background measurement, so it is dropped rather than counted,
            // matching the GPU kernel.
            if (value >= 0 && value < NUM_BG_BINS) {
                bins[static_cast<size_t>(value)] += static_cast<uint32_t>(count);
            } else if (value >= NUM_BG_BINS) {
                overflow_count += static_cast<uint32_t>(count);
            }
        }
    }

    ConstHistogramView view{bins.data(), NUM_BG_BINS, overflow_count};

    // Too many pixels in the overflow tail means NUM_BG_BINS is too small to
    // represent this reflection's background, so the estimate would diverge
    // from a full-range computation. Fail loudly rather than degrade silently.
    if (static_cast<double>(overflow_count)
        > kBackgroundMaxOverflowFraction * static_cast<double>(data.num_pixels())) {
        throw std::runtime_error(
          "Background histogram overflow exceeded the permitted fraction; "
          "NUM_BG_BINS is too small for this reflection's background level");
    }

    BackgroundResult result = tukey_constant_background(view);
    if (!result.valid) {
        throw std::runtime_error(
          "No background data remaining after outlier rejection");
    }

    return {result.mean, result.weighted_sum};
}
