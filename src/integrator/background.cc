/**
 * @file background.cc
 * @brief Host adapter from the baseline BackgroundAggregator histogram to the
 *        shared, device-safe constant background model.
 */

#include "integrator/background.hpp"

#include <cstdint>
#include <vector>

// Constant background estimate for one reflection. Delegates to the
// single-source model functions so the baseline and the GPU run identical math;
// this function only flattens the aggregator's array and overflow map into a
// contiguous ConstHistogramView over the shared NUM_BG_BINS range, then
// dispatches on the selected model. A rejected estimate returns a
// BackgroundResult with valid=false (the same channel the GPU reduction uses)
// so the caller can mark the reflection unintegrated rather than aborting.
BackgroundResult compute_background_constant_3d(const BackgroundAggregator &data,
                                                BackgroundModel model) {
    if (data.num_pixels() == 0) {
        return BackgroundResult{};  // no background pixels, so no estimate
    }

    const auto &small_hist = data.small_hist();
    const auto *large_hist = data.large_hist();  // may be nullptr

    // Bin into the shared [0, NUM_BG_BINS) range exactly as the GPU kernel does,
    // so the host baseline and the device reduction see identical histograms:
    // negative sentinel values are dropped, values at or above NUM_BG_BINS go to
    // the overflow tail.
    std::vector<uint32_t> bins(static_cast<size_t>(NUM_BG_BINS), 0);
    uint32_t in_range_count = 0;
    uint32_t overflow_count = 0;

    for (std::size_t v = 0; v < small_hist.size() && v < NUM_BG_BINS; ++v) {
        bins[v] = static_cast<uint32_t>(small_hist[v]);
        in_range_count += static_cast<uint32_t>(small_hist[v]);
    }
    if (large_hist != nullptr) {
        for (const auto &[value, count] : *large_hist) {
            // A negative value is a sentinel/garbage pixel, not a real
            // background measurement, so it is dropped rather than counted,
            // matching the GPU kernel.
            if (value >= 0 && value < NUM_BG_BINS) {
                bins[static_cast<size_t>(value)] += static_cast<uint32_t>(count);
                in_range_count += static_cast<uint32_t>(count);
            } else if (value >= NUM_BG_BINS) {
                overflow_count += static_cast<uint32_t>(count);
            }
        }
    }

    ConstHistogramView view{bins.data(), NUM_BG_BINS, overflow_count};

    // Too many pixels in the overflow tail means NUM_BG_BINS is too small to
    // represent this reflection's background, so the estimate would diverge
    // from a full-range computation. Reject it (the model functions apply the
    // same guard internally; this short-circuits with the explicit reason).
    // The denominator is the histogram population (in-range + overflow), which
    // matches the GPU reduction's count and excludes dropped sentinels.
    if (static_cast<double>(overflow_count)
        > kBackgroundMaxOverflowFraction
            * static_cast<double>(in_range_count + overflow_count)) {
        return BackgroundResult{};
    }

    // Dispatch to the selected single-source model, mirroring the GPU kernel.
    // The model functions already report rejection via BackgroundResult::valid.
    // No default case, so adding a BackgroundModel raises a -Wswitch warning here
    // until it is handled.
    switch (model) {
    case BackgroundModel::Constant:
        return tukey_constant_background(view);
    case BackgroundModel::Glm:
        return glm_constant_background(view);
    }

    // Unreachable: parse_background_model rejects unknown names. Keeps the
    // function total for an out-of-range model value.
    return BackgroundResult{};
}
