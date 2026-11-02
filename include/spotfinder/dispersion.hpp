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
 * @brief Which arithmetic form of the background test to evaluate.
 *
 * The two are the same comparison rearranged, the second multiplied through by
 * the mean so that the index of dispersion is never formed. They are not
 * bit-identical in float, and the GPU has always evaluated the first where the
 * CPU baseline evaluates the second, so the choice is explicit rather than
 * incidental.
 */
enum class DispersionImpl : uint8_t {
    GpuRatio,           ///< Form the index of dispersion, then compare
    DialsDivisionFree,  ///< Multiply the comparison through by the mean
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
 * @brief Apply the background and signal thresholds to one neighbourhood.
 *
 * The background comparison is formed from a sample mean and an unbiased
 * sample variance, and compared either as the index of dispersion against
 * 1 + n_sig_b·√(2/(n-1)) or with that bound multiplied back through by the
 * mean. The second spells the same inequality with one division fewer.
 *
 * Both follow DispersionThreshold::compute_threshold at
 * dials/algorithms/image/threshold/local.h:493-498 in DIALS 3.30, which is the
 * threshold dials.find_spots runs, though DIALS reaches it in double over
 * moments it also holds in double, where these are float.
 *
 * @param moments The neighbourhood's reduced moments.
 * @param min_count Fewest valid neighbours for either threshold to carry
 *                  meaning; below it the pixel is reported as neither.
 * @param n_sig_b Background threshold, in sampling deviations of the index of
 *                dispersion (σ of variance/μ).
 * @param n_sig_s Signal threshold, in Poisson deviations above the local mean
 *                (σ = √μ).
 * @param global_threshold Lowest count the centre pixel may have and still be
 *                         called signal, in PHOTONS. 0 disables the test,
 *                         which is what the GPU has always done.
 * @param impl Which arithmetic form to evaluate.
 * @return Both flags, each false when the neighbourhood is below min_count.
 */
FFS_HD inline DispersionFlags dispersion_flags(
  const DispersionMoments &moments,
  uint8_t min_count,
  float n_sig_b,
  float n_sig_s,
  float global_threshold = 0.0f,
  DispersionImpl impl = DispersionImpl::GpuRatio) {
    // Below min_count the sampling deviation is too wide for the comparison to
    // say anything, and a single valid pixel would divide by zero.
    if (moments.n < min_count) {
        return {false, false};
    }

    const float n = static_cast<float>(moments.n);
    const float sum = static_cast<float>(moments.sum);
    const float sumsq = static_cast<float>(moments.sumsq);

    // Sample mean, and the unbiased sample variance in the form that needs
    // only the two raw moments.
    const float mean = sum / n;
    const float variance = (n * sumsq - sum * sum) / (n * (n - 1.0f));

    const float background_threshold = 1.0f + n_sig_b * std::sqrt(2.0f / (n - 1.0f));
    bool not_background = false;
    switch (impl) {
    case DispersionImpl::GpuRatio:
        not_background = variance / mean > background_threshold;
        break;
    case DispersionImpl::DialsDivisionFree:
        not_background = variance > mean * background_threshold;
        break;
    }

    const float this_pixel = static_cast<float>(moments.this_pixel);
    const bool is_signal =
      this_pixel > mean + n_sig_s * std::sqrt(mean) && this_pixel >= global_threshold;

    return {not_background, is_signal};
}
