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
 *
 * The robust-Poisson GLM model and its symbols (η, μ, β, ψ_c, the score U and
 * the Fisher information I) follow Parkhurst, Winter, Waterman, Fuentes-Montero,
 * Gildea, Murshudov & Evans (2016), "Robust background modelling in DIALS",
 * J. Appl. Cryst. 49, 1912-1921, DOI 10.1107/S1600576716013595. Equation numbers
 * in the GLM comments below refer to that paper.
 */

#pragma once

#include <array>
#include <cmath>
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
 * DIALS "constant 3d" model). Glm is the DIALS robust-Poisson GLM constant
 * background (matches the DIALS "glm constant3d" model); it reads the same
 * histogram.
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

// Parameters for the robust-Poisson GLM constant background, matching the DIALS
// defaults (dials.algorithms.background.glm: tuning_constant 1.345,
// max_iter 100, tolerance 1e-3, min_pixels 10). Held here as the single source
// of truth so the host baseline and the device reduction iterate identically.
// kGlmTuningConstant is the Huber tuning constant c of ψ_c [Eq 3]; 1.345 gives
// 95% efficiency under a normal model.
constexpr double kGlmTuningConstant = 1.345;
constexpr double kGlmTolerance = 1e-3;
constexpr int kGlmMaxIter = 100;
constexpr uint32_t kGlmMinPixels = 10;

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
 * @brief Poisson probability mass P(Y = value).
 *
 * Used to form the GLM expectation values. value is integer-valued.
 *
 * Source: scitbx::glmtbx::poisson::pdf in scitbx/glmtbx/family.h.
 */
FFS_HD inline double glm_poisson_pdf(double mean, double value) {
    if (mean == 0.0) return 0.0;
    if (value == 0.0) return std::exp(-mean);
    if (value < 0.0) return 0.0;
    return std::exp(value * std::log(mean) - mean - std::lgamma(value + 1.0));
}

/**
 * @brief Poisson cumulative probability P(Y <= value) for integer value.
 *
 * The DIALS routine uses boost::math::gamma_q(floor(value+1), mean). For an
 * integer first argument that regularised upper incomplete gamma equals the
 * finite Poisson sum e^-mean * sum_{k=0..value} mean^k / k!, which avoids a
 * special-function dependency on the device. mean stays below NUM_BG_BINS for
 * accepted reflections, so the sum is short.
 *
 * Source: scitbx::glmtbx::poisson::cdf in scitbx/glmtbx/family.h.
 */
FFS_HD inline double glm_poisson_cdf(double mean, double value) {
    if (mean == 0.0) return 0.0;
    if (value < 0.0) return 0.0;
    const long v = static_cast<long>(std::floor(value));
    double term = std::exp(-mean);  // k = 0
    double sum = term;
    for (long k = 1; k <= v; ++k) {
        term *= mean / static_cast<double>(k);
        sum += term;
    }
    return sum;
}

/**
 * @brief Huber psi function ψ_c(r) [Eq 3]: identity for |r| < c, clipped to
 *        ±c outside.
 *
 * Source: scitbx::glmtbx::huber in scitbx/glmtbx/robust_glm.h.
 */
FFS_HD inline double glm_huber(double r, double c) {
    if (std::fabs(r) < c) return r;
    return (r > 0.0) ? c : ((r < 0.0) ? -c : 0.0);
}

/**
 * @brief Poisson expectation values used to centre and weight the robust score.
 *
 * epsi1 = E[ψ_c(rᵢ)], the per-observation expectation subtracted from ψ_c in
 * the score U [Eq 2]; weighted by μ′/√(φ*v_μ) it forms the consistency
 * correction a(β) [Eq 4]. epsi2 = E[ψ_c(rᵢ)*∂lnP(yᵢ|μ)/∂μ] (for Poisson
 * ∂lnP/∂μ = (yᵢ - μ)/v_μ), the expectation in the diagonal bᵢ of B [Eq 10,
 * Poisson form Eq 11]; B is the weight matrix of the Fisher information
 * I = XᵀBX [Eq 9].
 *
 * Source: the epsi1/epsi2 members of scitbx::glmtbx::expectation<poisson> in
 * scitbx/glmtbx/robust_glm.h.
 */
struct GlmExpectation {
    double epsi1 = 0.0;
    double epsi2 = 0.0;
};

/**
 * @brief Compute the Poisson expectation values E[ψ_c] (epsi1) and
 *        E[ψ_c*∂lnP/∂μ] (epsi2) for a given mean μ, sqrt-variance √(φ*v_μ) and
 *        Huber tuning constant c.
 *
 * The p1..p10 Poisson probabilities and the epsi1/epsi2 closed forms reproduce
 * the DIALS algebra.
 *
 * Source: the constructor of scitbx::glmtbx::expectation<poisson> in
 * scitbx/glmtbx/robust_glm.h.
 */
FFS_HD inline GlmExpectation glm_expectation(double mu, double svar, double c) {
    const double j1 = std::floor(mu - c * svar);
    const double j2 = std::floor(mu + c * svar);
    const double p1 = glm_poisson_pdf(mu, j1);        // P(Y  = j1)
    const double p2 = glm_poisson_pdf(mu, j2);        // P(Y  = j2)
    const double p3 = glm_poisson_cdf(mu, j1);        // P(Y <= j1)
    const double p4 = glm_poisson_pdf(mu, j2 + 1.0);  // P(Y  = j2 + 1)
    const double p5 = glm_poisson_cdf(mu, j2 + 1.0);  // P(Y <= j2 + 1)
    const double p6 = 1.0 - p5 + p4;                  // P(Y >= j2 + 1)
    const double p7 = glm_poisson_pdf(mu, j1 - 1.0);  // P(Y  = j1 - 1)
    const double p8 = glm_poisson_pdf(mu, j2 - 1.0);  // P(Y  = j2 - 1)
    const double p9 = glm_poisson_cdf(mu, j2 - 1.0);  // P(Y <= j2 - 1)
    const double p10 = p9 - p3 + p1;                  // P(j1 <= Y <= j2)

    GlmExpectation e;
    e.epsi1 = c * (p6 - p3) + (mu / svar) * (p1 - p2);
    e.epsi2 =
      c * (p1 + p2) + (mu * mu / (svar * svar * svar)) * (p10 / mu + p7 - p1 - p8 + p2);
    return e;
}

/**
 * @brief Robust-Poisson GLM constant background (DIALS "glm constant3d").
 *
 * Single-source implementation shared by host and device. Fits a constant
 * Poisson mean with a log link by iteratively reweighted least squares with
 * Huber weighting, reproducing dials::algorithms::RobustPoissonMean over the
 * same per-reflection histogram. Because every background pixel shares one
 * design row, the fit depends only on the value counts, so the histogram is an
 * exact representation. High-tail overflow pixels always sit far above the
 * upper Huber bound, so their psi clips to +c regardless of their exact value,
 * which is why the overflow count alone suffices.
 *
 * Iterating bins times their count and folding the overflow tail in at the
 * saturated Huber value are exact restatements of the DIALS per-pixel loop. The
 * sole numerical divergence is the Hessian: DIALS sums H += b per pixel while
 * this uses N * b directly, equal in exact arithmetic but differing in
 * floating-point rounding, so parity with DIALS holds to 1e-6 rather than
 * bit-for-bit (see the IRLS loop below).
 *
 * Paper symbols (constant model, design row xᵢ = (1)): coefficient β, linear
 * predictor η = β, mean μ = exp(η) (log link), link derivative μ′ = dμ/dη,
 * dispersion φ = 1 and variance function v_μ = μ, so √(φ*v_μ) = √μ. The IRLS
 * loop below forms the robust score U [Eq 2] and the Fisher information I [Eq 9]
 * and applies the update β ← β + I⁻¹U [Eq 5].
 *
 * The seed is the histogram median (matching DIALS), the tuning constant,
 * tolerance, iteration cap and minimum pixel count match the DIALS defaults.
 * Device-safe: no allocation, no exceptions; failure (too few pixels, range too
 * small, non-convergence, or a degenerate parameter) is reported via
 * BackgroundResult::valid. weighted_sum is the modelled background summed over
 * the background pixels (mean * N), since the GLM models every pixel at mean.
 *
 * Source: dials::algorithms::RobustPoissonMean in
 * dials/algorithms/background/glm/robust_poisson_mean.h (the constant-model
 * specialisation of scitbx::glmtbx::robust_glm in scitbx/glmtbx/robust_glm.h).
 */
FFS_HD inline BackgroundResult glm_constant_background(const ConstHistogramView &hist) {
    BackgroundResult result;

    // Total pixel count across the histogram and the high-tail overflow.
    uint64_t N = hist.overflow_count;
    for (int v = 0; v < hist.num_bins; ++v) {
        N += hist.bins[v];
    }
    // DIALS requires at least min_pixels background pixels to attempt a fit.
    if (N < kGlmMinPixels) {
        return result;
    }

    // Too much of the background in the high-tail overflow means the histogram
    // range is too small to characterise this reflection (see
    // tukey_constant_background); reject rather than fit a truncated histogram.
    if (static_cast<double>(hist.overflow_count)
        > kBackgroundMaxOverflowFraction * static_cast<double>(N)) {
        return result;
    }

    // Median seed, matching DIALS detail::median (the element at sorted
    // position N/2). The overflow tail counts towards the position but, for an
    // accepted reflection, the median itself lies within the binned range.
    const uint64_t mid = N / 2;  // 0-based target index
    uint64_t cumulative = 0;
    long median = -1;
    for (int v = 0; v < hist.num_bins; ++v) {
        cumulative += hist.bins[v];
        if (cumulative >= mid + 1) {
            median = v;
            break;
        }
    }
    double mean0 = (median < 0) ? 1.0 : static_cast<double>(median);
    if (mean0 == 0.0) mean0 = 1.0;  // DIALS: a zero median seeds at 1

    // IRLS for the single coefficient β = log(μ) (constant model, log link).
    const double c = kGlmTuningConstant;
    double beta = std::log(mean0);
    std::size_t niter = 0;
    for (niter = 0; niter < static_cast<std::size_t>(kGlmMaxIter); ++niter) {
        const double eta = beta;           // η = β
        const double mu = std::exp(eta);   // μ = exp(η), linkinv
        const double dmu = std::exp(eta);  // μ′ = dμ/dη, dmu/deta
        const double svar =
          std::sqrt(mu);  // √(φ*v_μ), φ = 1, v_μ = μ, sqrt(phi * variance), phi = 1
        if (!(mu > 0.0) || !(svar > 0.0)) {
            return result;  // degenerate, cannot continue (valid stays false)
        }

        const GlmExpectation epsi = glm_expectation(mu, svar, c);
        // bᵢ = epsi2*μ′²/√(φ*v_μ), the diagonal of B [Eq 10], where the
        // expectation factor epsi2 = E[ψ_c*∂lnP/∂μ] (Poisson form, Eq 11);
        // constant across observations here (w = 1).
        const double b = epsi.epsi2 * dmu * dmu / svar;

        // Robust score U = Σ (ψ_c(rᵢ) - E[ψ_c])*μ′/√(φ*v_μ) [Eq 2], with the
        // Pearson residual rᵢ = (yᵢ - μ)/√v_μ and E[ψ_c] = epsi1. Expanding the
        // subtraction recovers the paper's per-term form
        // Σ (ψ_c(rᵢ)*μ′/√(φ*v_μ) - a(β)), where the consistency correction
        // a(β) = E[ψ_c]*μ′/√(φ*v_μ) [Eq 4] (μ′ and √(φ*v_μ) are constant across
        // observations). Each bin
        // contributes its count times the per-value term; the overflow pixels
        // are extreme high outliers whose ψ_c clips to +c, so they contribute
        // the same term regardless of their exact (unrecorded) value.
        double U = 0.0;
        for (int v = 0; v < hist.num_bins; ++v) {
            const uint32_t count = hist.bins[v];
            if (count == 0) {
                continue;
            }
            const double res = (static_cast<double>(v) - mu) / svar;  // rᵢ
            const double q = (glm_huber(res, c) - epsi.epsi1) * dmu / svar;
            U += static_cast<double>(count) * q;
        }
        if (hist.overflow_count > 0) {
            const double q = (c - epsi.epsi1) * dmu / svar;
            U += static_cast<double>(hist.overflow_count) * q;
        }

        // Fisher information I = XᵀBX [Eq 9], a scalar N*b for the constant
        // model. DIALS accumulates H += b once per observation, so H = n_obs*b;
        // since b is constant here this folds to N*b directly. The result is
        // identical in exact arithmetic; the only divergence from DIALS is
        // floating-point rounding (N*b versus summing b N times), which holds
        // the histogram path to 1e-6 parity rather than bit-for-bit equality.
        // delta = I⁻¹U, and beta += delta is the IRLS update β <- β + I⁻¹U [Eq 5].
        const double delta = U / (static_cast<double>(N) * b);
        const double sum_delta_sq = delta * delta;
        const double sum_beta_sq = beta * beta;
        beta += delta;

        const double error =
          std::sqrt(sum_delta_sq / (sum_beta_sq > 1e-10 ? sum_beta_sq : 1e-10));
        if (error < kGlmTolerance) {
            break;
        }
    }

    // DIALS treats a run that exhausts max_iter as non-converged and fails the
    // reflection; mirror that, and the mean()'s bound on beta.
    if (niter >= static_cast<std::size_t>(kGlmMaxIter)) {
        return result;
    }
    if (!(beta > -300.0 && beta < 300.0)) {
        return result;
    }

    const double mean = std::exp(beta);  // μ = exp(β)
    result.mean = mean;
    result.weighted_sum = mean * static_cast<double>(N);
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
        // Negatives are garbage pixels that slipped past the mask, not real
        // background measurements, so they are dropped entirely.
        if (x < 0) {
            return;
        }
        if (x < VECTOR_LIMIT) {
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
 * @brief Estimate a constant background level from an aggregated histogram.
 *
 * Flattens the aggregator into the shared ConstHistogramView and dispatches to
 * the selected single-source model: tukey_constant_background (Constant) or
 * glm_constant_background (Glm), so the baseline runs the same math as the GPU.
 *
 * @param data Aggregated background pixel histogram for one reflection.
 * @param impl Which implementation to run: the independent dials-like baseline
 *        (default) or the shared core the GPU uses. The dials-like baseline is
 *        Tukey-only and ignores model.
 * @param model Background model the shared core applies (Constant = Tukey;
 *        Glm = robust-Poisson GLM).
 * @return BackgroundResult with mean and weighted_sum; valid is false when the
 *         estimate is rejected (no inliers, too few pixels, too much overflow,
 *         or non-convergence), in which case the caller marks the reflection
 *         unintegrated. Mirrors the BackgroundResult::valid channel the GPU
 *         reduction uses.
 */
BackgroundResult compute_background_constant_3d(
  const BackgroundAggregator &data,
  ConstantBackgroundImpl impl = ConstantBackgroundImpl::DialsIndependent,
  BackgroundModel model = BackgroundModel::Constant);
