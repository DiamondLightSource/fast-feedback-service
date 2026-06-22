/**
 * @file background.cc
 * @brief Host constant background estimation for the baseline integrator,
 *        available as either the independent dials-like algorithm or an adapter
 *        onto the shared, device-safe constant background models (Tukey/GLM).
 */

#include "integrator/background.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

// Independent dials-like constant background. Self-contained baseline that
// scans the aggregator's unbounded histogram directly: the small fixed array
// for low values and, only if a quartile is not yet found, the sparse map of
// large/outlier values. Every pixel is counted (including negative sentinels
// in the large map) and there is no overflow rejection, so this stays the
// true-to-dials reference independent of the shared core's bounded range. This
// path is Tukey/IQR only; the GLM model lives solely in the shared core. A
// degenerate input returns a BackgroundResult with valid=false (the same
// channel the GPU reduction uses) so the caller can skip the reflection rather
// than aborting.
BackgroundResult compute_background_constant_3d_dials(
  const BackgroundAggregator &data) {
    constexpr double iqr_multiplier = 1.5;

    const int N = data.num_pixels();
    if (N == 0) {
        return BackgroundResult{};  // no background pixels, so no estimate
    }

    // Quantile positions (1-based counting convention)
    const std::size_t p25 = (N + 3) / 4;
    const std::size_t p50 = (N + 1) / 2;
    const std::size_t p75 = (3 * N + 1) / 4;

    const auto &small_hist = data.small_hist();
    const auto *large_hist = data.large_hist();  // may be nullptr

    std::size_t cumulative = 0;
    int q1 = -1, median = -1, q3 = -1;

    // ---- Scan small histogram (fixed array) ----
    for (std::size_t value = 0; value < small_hist.size(); ++value) {
        cumulative += small_hist[value];

        if (q1 < 0 && cumulative >= p25) q1 = static_cast<int>(value);
        if (median < 0 && cumulative >= p50) median = static_cast<int>(value);
        if (q3 < 0 && cumulative >= p75) {
            q3 = static_cast<int>(value);
            break;
        }
    }

    // ---- Scan large histogram only if needed ----
    if (q3 < 0 && large_hist != nullptr) {
        std::vector<int> keys;
        keys.reserve(large_hist->size());
        for (const auto &[k, _] : *large_hist) {
            keys.push_back(k);
        }
        std::sort(keys.begin(), keys.end());

        for (int value : keys) {
            cumulative += large_hist->at(value);

            if (q1 < 0 && cumulative >= p25) q1 = value;
            if (median < 0 && cumulative >= p50) median = value;
            if (q3 < 0 && cumulative >= p75) {
                q3 = value;
                break;
            }
        }
    }

    // Quartiles not found means the input is inconsistent; reject the estimate.
    if (q1 < 0 || q3 < 0) {
        return BackgroundResult{};
    }

    const int iqr = q3 - q1;
    const double lower_bound = q1 - iqr_multiplier * iqr;
    const double upper_bound = q3 + iqr_multiplier * iqr;

    // ---- Accumulate inliers ----
    std::size_t included_count = 0;
    double weighted_sum = 0.0;

    // Small histogram
    for (std::size_t value = 0; value < small_hist.size(); ++value) {
        if (value < lower_bound || value > upper_bound) {
            continue;
        }

        const std::size_t count = small_hist[value];
        included_count += count;
        weighted_sum += static_cast<double>(value) * count;
    }

    // Large histogram (if present)
    if (large_hist != nullptr) {
        for (const auto &[value, count] : *large_hist) {
            if (value < lower_bound || value > upper_bound) {
                continue;
            }

            included_count += count;
            weighted_sum += static_cast<double>(value) * count;
        }
    }

    if (included_count == 0) {
        return BackgroundResult{};  // no inliers after outlier rejection
    }

    const double mean = weighted_sum / static_cast<double>(included_count);
    return BackgroundResult{mean, weighted_sum, true};
}

// Shared-core constant background. Delegates to the single-source model
// functions so the baseline and the GPU run identical math; this function only
// flattens the aggregator's array and overflow map into a contiguous
// ConstHistogramView over the shared NUM_BG_BINS range, then dispatches on the
// selected model. A rejected estimate returns a BackgroundResult with
// valid=false (the same channel the GPU reduction uses) so the caller can mark
// the reflection unintegrated rather than aborting.
BackgroundResult compute_background_constant_3d_shared(const BackgroundAggregator &data,
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

}  // namespace

BackgroundResult compute_background_constant_3d(const BackgroundAggregator &data,
                                                ConstantBackgroundImpl impl,
                                                BackgroundModel model) {
    switch (impl) {
    case ConstantBackgroundImpl::SharedCore:
        return compute_background_constant_3d_shared(data, model);
    case ConstantBackgroundImpl::DialsIndependent:
    default:
        // The dials reference is Tukey-only; the model selection is ignored.
        return compute_background_constant_3d_dials(data);
    }
}
