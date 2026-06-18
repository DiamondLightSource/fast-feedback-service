/**
 * @file background.cc
 * @brief Host constant (Tukey/IQR) background estimation for the baseline
 *        integrator, available as either the independent dials-like algorithm
 *        or an adapter onto the shared, device-safe constant background model.
 */

#include "integrator/background.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

// Independent dials-like constant background. Self-contained baseline that
// scans the aggregator's unbounded histogram directly: the small fixed array
// for low values and, only if a quartile is not yet found, the sparse map of
// large/outlier values. Every pixel is counted (including negative sentinels
// in the large map) and there is no overflow rejection, so this stays the
// true-to-dials reference independent of the shared core's bounded range.
std::tuple<double, double> compute_background_constant_3d_dials(
  const BackgroundAggregator &data) {
    constexpr double iqr_multiplier = 1.5;

    const int N = data.num_pixels();
    if (N == 0) {
        throw std::runtime_error("No background pixels available");
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

    // Sanity check (should not happen unless input is inconsistent)
    if (q1 < 0 || q3 < 0) {
        throw std::runtime_error("Failed to compute quartiles for background");
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
        throw std::runtime_error(
          "No background data remaining after outlier rejection");
    }

    const double mean = weighted_sum / static_cast<double>(included_count);
    return {mean, weighted_sum};
}

// Shared-core constant background. Delegates to the single-source
// tukey_constant_background() so the baseline and the GPU run identical math;
// this function only flattens the aggregator's array and overflow map into a
// contiguous ConstHistogramView over the shared NUM_BG_BINS range.
std::tuple<double, double> compute_background_constant_3d_shared(
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
    // from a full-range computation. Fail loudly rather than degrade silently.
    // The denominator is the histogram population (in-range + overflow), which
    // matches the GPU reduction's count and excludes dropped sentinels.
    if (static_cast<double>(overflow_count)
        > kBackgroundMaxOverflowFraction
            * static_cast<double>(in_range_count + overflow_count)) {
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

}  // namespace

std::tuple<double, double> compute_background_constant_3d(
  const BackgroundAggregator &data,
  ConstantBackgroundImpl impl) {
    switch (impl) {
    case ConstantBackgroundImpl::SharedCore:
        return compute_background_constant_3d_shared(data);
    case ConstantBackgroundImpl::DialsIndependent:
    default:
        return compute_background_constant_3d_dials(data);
    }
}
