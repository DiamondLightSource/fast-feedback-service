/**
 * @file dispersion.hpp
 * @brief The dispersion thresholding decision, shared by the GPU kernels and
 *        the CPU baseline.
 *
 * Background on a photon-counting detector is Poisson, so a patch of pure
 * background has a variance equal to its mean. The ratio of the two is 1 for
 * background and rises where a spot contributes; a pixel is strong when its
 * patch fails that test and the pixel itself stands clear of the local mean.
 *
 * Callers do their own accumulation, by shared-memory window on the GPU and by
 * summed-area table in the baseline, and hand in the result. The moments are
 * integers so that a bright neighbourhood sums without rounding, which assumes
 * whole photon counts; a detector with non-integer values, Jungfrau among them,
 * requires changes here.
 */

#pragma once

#include <cmath>
#include <cstdint>

#include "ffs_hd.hpp"

/**
 * @brief Which arithmetic form of the thresholds to evaluate.
 *
 * The two are the same comparison rearranged, and disagree on pixels sitting
 * within a float ulp of either bound, so the choice is explicit rather than
 * incidental.
 */
enum class DispersionImpl : uint8_t {
    FastInteger,        ///< Multiply through to whole numbers, as DIALS does
    DialsDivisionFree,  ///< Form the mean and variance, then compare in float
};

/**
 * @brief One pixel's neighbourhood, reduced to what the decision needs.
 *
 * Counts are in photons, so `sumsq` is in photons². `this_pixel` is held at 32
 * bits rather than at `pixel_t` so that one definition covers both compiled
 * pixel widths and this header stays free of the reader's headers.
 */
struct DispersionMoments {
    uint64_t sum;         ///< Σ of the valid neighbouring counts
    uint64_t sumsq;       ///< Σ of their squares
    uint32_t this_pixel;  ///< The centre pixel's own count
    uint8_t n;            ///< Valid pixels in the neighbourhood, centre included
};

/**
 * @brief The outcome of the two thresholds for one pixel.
 */
struct DispersionFlags {
    bool not_background;  ///< The neighbourhood is more variable than Poisson
    bool is_signal;       ///< The centre pixel stands above the local mean
};

/**
 * @brief The two arithmetic forms, reached only through dispersion_flags.
 *
 * Both take a neighbourhood already known to hold at least min_count valid
 * pixels; neither is safe at n = 1, where the sampling deviation divides by
 * zero. Parameters carry the meanings documented on dispersion_flags.
 */
namespace dispersion_detail {

/**
 * @brief Evaluate both thresholds in float, through a mean and a variance.
 */
FFS_HD inline DispersionFlags division_free(const DispersionMoments &moments,
                                            float n_sig_b,
                                            float n_sig_s,
                                            float global_threshold) {
    const float n = static_cast<float>(moments.n);
    const float sum = static_cast<float>(moments.sum);
    const float sumsq = static_cast<float>(moments.sumsq);

    // Sample mean, and the unbiased sample variance in the form that needs
    // only the two raw moments.
    const float mean = sum / n;
    const float variance = (n * sumsq - sum * sum) / (n * (n - 1.0f));

    const float background_threshold = 1.0f + n_sig_b * std::sqrt(2.0f / (n - 1.0f));
    const float this_pixel = static_cast<float>(moments.this_pixel);

    return {
      variance > mean * background_threshold,
      this_pixel > mean + n_sig_s * std::sqrt(mean) && this_pixel >= global_threshold};
}

/**
 * @brief Evaluate both thresholds in 64-bit integers, on the raw moments.
 */
FFS_HD inline DispersionFlags fast_integer(const DispersionMoments &moments,
                                           float n_sig_b,
                                           float n_sig_s,
                                           float global_threshold) {
    // The subtraction below cancels almost entirely over a background patch,
    // so it runs on the exact counts rather than in float.
    const int64_t n = static_cast<int64_t>(moments.n);
    const int64_t sum = static_cast<int64_t>(moments.sum);
    const int64_t sumsq = static_cast<int64_t>(moments.sumsq);
    const int64_t this_pixel = static_cast<int64_t>(moments.this_pixel);

    // How far the variance runs beyond the mean, against what n_sig_b allows.
    // Both sides carry the same scale, so neither is divided back down.
    const int64_t variance_excess = n * sumsq - sum * sum - sum * (n - 1);
    const float variance_allowance =
      static_cast<float>(sum) * n_sig_b * std::sqrt(2.0f * static_cast<float>(n - 1));

    // The same comparison for the centre pixel against the local mean.
    const int64_t pixel_excess = n * this_pixel - sum;
    const float pixel_allowance =
      n_sig_s * std::sqrt(static_cast<float>(sum) * static_cast<float>(n));

    const bool is_signal = static_cast<float>(pixel_excess) > pixel_allowance
                           && static_cast<float>(this_pixel) >= global_threshold;

    return {static_cast<float>(variance_excess) > variance_allowance, is_signal};
}

}  // namespace dispersion_detail

/**
 * @brief Apply the background and signal thresholds to one neighbourhood.
 *
 * Reproduces DispersionThreshold::compute_threshold at
 * dials/algorithms/image/threshold/local.h:493-498 in DIALS 3.30.
 *
 * @param moments The neighbourhood's reduced moments.
 * @param min_count Fewest valid neighbours for either threshold to carry
 *                  meaning; below it the pixel is reported as neither.
 * @param n_sig_b Background threshold, in sampling deviations of the index of
 *                dispersion (σ of variance/μ).
 * @param n_sig_s Signal threshold, in Poisson deviations above the local mean
 *                (σ = √μ).
 * @param global_threshold Lowest count the centre pixel may have and still be
 *                         called signal, in PHOTONS. 0 disables the test.
 * @param impl Which arithmetic form to evaluate.
 * @return Both flags, each false when the neighbourhood is below min_count.
 */
FFS_HD inline DispersionFlags dispersion_flags(
  const DispersionMoments &moments,
  uint8_t min_count,
  float n_sig_b,
  float n_sig_s,
  float global_threshold = 0.0f,
  DispersionImpl impl = DispersionImpl::FastInteger) {
    // Below min_count the sampling deviation is too wide for the comparison to
    // say anything, and a single valid pixel leaves no allowance at all.
    if (moments.n < min_count) {
        return {false, false};
    }

    switch (impl) {
    case DispersionImpl::DialsDivisionFree:
        return dispersion_detail::division_free(
          moments, n_sig_b, n_sig_s, global_threshold);
    case DispersionImpl::FastInteger:
        return dispersion_detail::fast_integer(
          moments, n_sig_b, n_sig_s, global_threshold);
    }

    return {false, false};
}
