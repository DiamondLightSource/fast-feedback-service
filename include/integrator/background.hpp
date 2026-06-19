/**
 * @file background.hpp
 * @brief Per-reflection background estimation, shared by the baseline CPU
 *        integrator and the GPU integrator.
 *
 * The constant (Tukey/IQR) background model is implemented once, as a
 * device-safe function over a uniform integer-histogram view
 * (::ConstHistogramView). The same code compiles for the host (baseline) and
 * for CUDA device code (GPU reduction kernel), so both paths produce identical
 * results. Background pixel values are integer counts, so a histogram with one
 * bin per integer value makes the quartile/IQR logic exact.
 *
 * This assumes a photon-counting detector, where raw pixel values are
 * non-negative integer photon counts. The integer histogram and the dropping
 * of negative values both depend on that assumption. Charge-integrating
 * detectors (e.g. Jungfrau) produce non-integer pixel values that will 
 * need a different background approach.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_map>

// __host__ __device__ under nvcc, nothing under a plain C++ compiler, so this
// header compiles in the baseline's CXX translation unit as well as in CUDA.
#if defined(__CUDACC__)
#define FFS_HD __host__ __device__
#else
#define FFS_HD
#endif

/**
 * @brief Selects which background model is used to estimate the per-reflection
 *        background level from its pixel histogram.
 *
 * Constant is the Tukey/IQR outlier-rejecting constant background (matches the
 * DIALS "constant 3d" model). Glm is the full DIALS robust-Poisson GLM model,
 * not yet implemented; it will read the same histogram.
 */
enum class BackgroundModel : uint8_t { Constant, Glm };

/**
 * @brief Selects which implementation of the constant (Tukey/IQR) background
 *        model the host baseline uses.
 *
 * DialsIndependent is the original self-contained baseline: an unbounded
 * histogram (small array plus a sparse map for large/outlier values) that
 * counts every pixel, including negative sentinels, with no overflow rejection.
 * This is the true-to-dials reference. SharedCore delegates to the same
 * tukey_constant_background() the GPU runs, over a bounded NUM_BG_BINS
 * histogram, so the baseline can be compared directly against the shared core.
 */
enum class ConstantBackgroundImpl : uint8_t { DialsIndependent, SharedCore };

// Number of integer-valued bins in each per-reflection background histogram.
// Single source of truth, shared by the GPU device histogram stride and the
// host baseline adapter, so both paths bin identical pixel values and produce
// identical estimates. bins cover pixel values [0, NUM_BG_BINS); values at or
// above NUM_BG_BINS land in the per-reflection overflow tail. Chosen above the
// realistic constant-background inlier range; device-memory cost is
// num_reflections * NUM_BG_BINS * 4 bytes. If a reflection puts more than
// kBackgroundMaxOverflowFraction of its background pixels in the overflow, the
// range is too small to characterise that background and the estimate is
// rejected (see tukey_constant_background), which the callers surface as an
// error rather than silently degrading.
constexpr int NUM_BG_BINS = 256;

// Fraction of a reflection's background pixels allowed in the high-tail
// overflow before the constant background estimate is rejected as untrustworthy.
constexpr double kBackgroundMaxOverflowFraction = 0.25;

/**
 * @brief Read-only view of a per-reflection background histogram.
 *
 * bins[v] holds the number of background pixels with integer value v, for
 * v in [0, num_bins). overflow_count holds the number of pixels with value
 * >= num_bins (the high tail). For a constant background, num_bins is chosen
 * above the realistic inlier range, so the overflow only ever contains high
 * outliers, which Tukey rejects; this keeps the bounded histogram exact.
 */
struct ConstHistogramView {
    const uint32_t *bins = nullptr;
    int num_bins = 0;
    uint32_t overflow_count = 0;
};

/**
 * @brief Result of a constant background estimate.
 *
 * mean is the background level per pixel; weighted_sum is the sum of the
 * inlier pixel values used to form it (DIALS background.sum.value). valid is
 * false when there are no pixels or no inliers survive outlier rejection.
 */
struct BackgroundResult {
    double mean = 0.0;
    double weighted_sum = 0.0;
    bool valid = false;
};

/**
 * @brief Tukey (IQR-based) outlier-rejecting constant background.
 *
 * Single-source implementation shared by host and device. Computes the
 * quartiles of the histogram, rejects values outside
 * [q1 - 1.5*IQR, q3 + 1.5*IQR], and returns the mean and weighted sum of the
 * surviving inliers. Device-safe: no allocation, no exceptions; failure is
 * reported via BackgroundResult::valid.
 *
 * The high tail (overflow_count) is counted towards the quartile positions but
 * never contributes inliers, since for a sensible num_bins it lies above the
 * upper rejection bound.
 */
FFS_HD inline BackgroundResult tukey_constant_background(
  const ConstHistogramView &hist) {
    constexpr double iqr_multiplier = 1.5;

    // Defaults to valid=false; set true only once a mean has been computed from
    // real inliers at the end.
    BackgroundResult result;

    // Total pixel count across the histogram and the high-tail overflow.
    uint64_t N = hist.overflow_count;
    for (int v = 0; v < hist.num_bins; ++v) {
        N += hist.bins[v];
    }
    if (N == 0) {
        return result;  // no background pixels, so no estimate (valid stays false)
    }

    // Too much of the background in the high-tail overflow means the histogram
    // range (num_bins) is too small to characterise this reflection. The
    // quartile and inlier estimate would silently diverge from a full-range
    // computation, so reject it (valid stays false) and let the caller report
    // the insufficient range.
    if (static_cast<double>(hist.overflow_count)
        > kBackgroundMaxOverflowFraction * static_cast<double>(N)) {
        return result;
    }

    // Quantile positions (1-based counting convention, matching the baseline).
    const uint64_t p25 = (N + 3) / 4;
    const uint64_t p50 = (N + 1) / 2;
    const uint64_t p75 = (3 * N + 1) / 4;

    // Ascending scan over bins to locate q1, median, q3. If a quartile
    // falls in the overflow tail, num_bins is too small for this
    // reflection; clamp it to num_bins (a lower bound). A clamped q1
    // alone leaves the estimate usable, whereas a clamped q3 pushes the
    // upper fence to num_bins, which the bound check below rejects.
    uint64_t cumulative = 0;
    long q1 = -1, median = -1, q3 = -1;
    for (int v = 0; v < hist.num_bins; ++v) {
        cumulative += hist.bins[v];
        if (q1 < 0 && cumulative >= p25) q1 = v;
        if (median < 0 && cumulative >= p50) median = v;
        if (q3 < 0 && cumulative >= p75) {
            q3 = v;
            break;
        }
    }
    if (q1 < 0) q1 = hist.num_bins;
    if (q3 < 0) q3 = hist.num_bins;

    const double iqr = static_cast<double>(q3 - q1);
    const double lower_bound = q1 - iqr_multiplier * iqr;
    const double upper_bound = q3 + iqr_multiplier * iqr;

    // The upper fence reaching past the bins into the overflow tail
    // means the range is too small to separate the inliers from the
    // high tail, so reject the estimate (valid stays false).
    if (upper_bound >= static_cast<double>(hist.num_bins)) {
        return result;
    }

    // Accumulate inliers. The fence check above guarantees
    // upper_bound < num_bins, so overflow values (>= num_bins)
    // always sit above the upper bound and are rejected
    uint64_t included_count = 0;
    double weighted_sum = 0.0;
    for (int v = 0; v < hist.num_bins; ++v) {
        if (v < lower_bound || v > upper_bound) continue;
        const uint64_t count = hist.bins[v];
        included_count += count;
        weighted_sum += static_cast<double>(v) * static_cast<double>(count);
    }

    if (included_count == 0) {
        return result;  // every pixel rejected as an outlier (valid stays false)
    }

    result.mean = weighted_sum / static_cast<double>(included_count);
    result.weighted_sum = weighted_sum;
    result.valid = true;
    return result;
}

/**
 * @brief Accumulates a histogram of background pixel values for a single
 * reflection so that a robust constant background can be estimated.
 *
 * Two data structures back the histogram:
 *  - a small fixed array for low values (< VECTOR_LIMIT), which is the vast
 *    majority of pixels and is efficient for adding many low-value pixels;
 *  - a lazily-allocated unordered map for large/sparse values (outliers).
 */
class BackgroundAggregator {
  public:
    BackgroundAggregator() = default;

    ~BackgroundAggregator() {
        delete _large_hist;
    }

    void add(int x) {
        if (x >= 0 && x < VECTOR_LIMIT) {
            ++_small_hist[x];
        } else {
            if (!_large_hist) {
                _large_hist = new std::unordered_map<int, std::size_t>();
            }
            ++(*_large_hist)[x];
        }
        ++n_pixels;
    }

    int num_pixels() const {
        return n_pixels;
    }
    const auto &small_hist() const {
        return _small_hist;
    }
    const auto *large_hist() const {
        return _large_hist;
    }

    void add(const BackgroundAggregator &other) {
        for (std::size_t i = 0; i < VECTOR_LIMIT; ++i) {
            _small_hist[i] += other._small_hist[i];
        }

        if (other._large_hist) {
            if (!_large_hist) {
                _large_hist = new std::unordered_map<int, std::size_t>();
            }
            for (const auto &[k, v] : *other._large_hist) {
                (*_large_hist)[k] += v;
            }
        }

        n_pixels += other.n_pixels;
    }

  private:
    static constexpr std::size_t VECTOR_LIMIT = 64;

    std::array<std::size_t, VECTOR_LIMIT> _small_hist{};
    std::unordered_map<int, std::size_t> *_large_hist = nullptr;
    int n_pixels = 0;
};

/**
 * @brief Estimate a constant background level from an aggregated histogram
 * using a Tukey (IQR-based) outlier rejection.
 *
 * @param data Aggregated background pixel histogram for one reflection.
 * @param impl Which implementation to run: the independent dials-like baseline
 *        (default) or the shared core the GPU uses.
 * @return {mean background, weighted sum of included pixel values}
 */
std::tuple<double, double> compute_background_constant_3d(
  const BackgroundAggregator &data,
  ConstantBackgroundImpl impl = ConstantBackgroundImpl::DialsIndependent);
