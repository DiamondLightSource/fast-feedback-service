/**
 * @file background.hpp
 * @brief Per-reflection background estimation, shared by the baseline CPU
 *        integrator and the GPU integrator.
 *
 * The constant (Tukey/IQR) background model is implemented once, as a
 * device-safe function over a sparse histogram view (::SparseHistogramView).
 * The same code compiles for the host (baseline) and for CUDA device code (GPU
 * reduction kernel), so both paths produce identical results. Background pixel
 * values are integer counts, so one entry per distinct value makes the
 * quartile/IQR logic exact for any value.
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
 * tukey_constant_background() the GPU runs, over the same sparse histogram, so
 * the baseline can be compared directly against the shared core.
 */
enum class ConstantBackgroundImpl : uint8_t { DialsIndependent, SharedCore };

// Number of open-addressed slots in each per-reflection sparse background
// histogram. This bounds the number of DISTINCT pixel values a reflection may
// hold, not the values themselves, so any 32-bit pixel value is representable
// exactly and there is no high tail. Measured
// occupancy is a median of 2-3 distinct values with a worst case of 39, so 64
// leaves substantial headroom; device-memory cost is
// num_reflections * NUM_BG_SLOTS * 8 bytes. A reflection that exceeds this
// spills, and a spilled reflection is rejected (see
// tukey_constant_background).
constexpr int NUM_BG_SLOTS = 64;

static_assert((NUM_BG_SLOTS & (NUM_BG_SLOTS - 1)) == 0,
              "NUM_BG_SLOTS must be a power of two so the probe index can be "
              "masked rather than reduced modulo");

/// @brief log2(NUM_BG_SLOTS), the shift a hashed value is reduced by.
FFS_HD constexpr int background_slot_bits() {
    int bits = 0;
    for (int n = NUM_BG_SLOTS; n > 1; n >>= 1) {
        ++bits;
    }
    return bits;
}

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
 * @brief Read-only view of a per-reflection sparse background histogram.
 *
 * entries holds one packed (value, count) pair per distinct pixel value,
 * sorted ascending by value; entries with a zero count are not present.
 * spill_count holds the number of pixels dropped because the slot table that
 * produced these entries was full, which makes the histogram incomplete by an
 * unknown amount and so fails the reflection.
 *
 * Ascending order is a precondition: both models locate quartiles by a
 * cumulative scan and rely on it.
 */
struct SparseHistogramView {
    const unsigned long long *entries = nullptr;
    int num_entries = 0;
    uint32_t spill_count = 0;
};

/**
 * @brief Pack a (value, count) pair into one 64-bit histogram entry.
 *
 * The value occupies the high word and the count the low word, so a count can
 * be incremented with a single 64-bit add without disturbing the value. That
 * holds because a count is bounded by the number of background pixels in one
 * reflection (tens of thousands) and so can never carry into the high word.
 * A stored entry always has a count of at least 1, which makes the all-zero
 * word an unambiguous empty marker.
 */
FFS_HD inline unsigned long long background_entry_pack(uint32_t value, uint32_t count) {
    return (static_cast<unsigned long long>(value) << 32)
           | static_cast<unsigned long long>(count);
}

/// @brief Pixel value of a packed histogram entry.
FFS_HD inline uint32_t background_entry_value(unsigned long long entry) {
    return static_cast<uint32_t>(entry >> 32);
}

/// @brief Pixel count of a packed histogram entry.
FFS_HD inline uint32_t background_entry_count(unsigned long long entry) {
    return static_cast<uint32_t>(entry & 0xFFFFFFFFull);
}

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
 * @brief Tukey (IQR-based) outlier-rejecting constant background over a sparse
 *        histogram.
 *
 * Single-source implementation shared by host and device. Computes the
 * quartiles of the histogram, rejects values outside
 * [q1 - 1.5*IQR, q3 + 1.5*IQR], and returns the mean and weighted sum of the
 * surviving inliers. Failure is reported via BackgroundResult::valid.
 *
 * Entries carry their own values, so there is no representable range and no
 * high tail: any pixel value is held exactly. What a slot table can exhaust
 * instead is the number of distinct values, and a full table (spill_count > 0)
 * has lost pixels of unknown magnitude, so the estimate is rejected rather
 * than computed from a truncated histogram.
 */
FFS_HD inline BackgroundResult tukey_constant_background(
  const SparseHistogramView &hist) {
    constexpr double iqr_multiplier = 1.5;

    // Defaults to valid=false; set true only once a mean has been computed from
    // real inliers at the end.
    BackgroundResult result;

    // Total pixel count across the histogram.
    uint64_t N = 0;
    for (int i = 0; i < hist.num_entries; ++i) {
        N += background_entry_count(hist.entries[i]);
    }
    if (N == 0) {
        return result;  // no background pixels, so no estimate (valid stays false)
    }

    // Pixels were dropped by a full slot table, so the histogram is incomplete
    // by an unknown amount and the quartiles cannot be trusted.
    if (hist.spill_count > 0) {
        return result;
    }

    // Quantile positions (1-based counting convention, matching the baseline).
    const uint64_t p25 = (N + 3) / 4;
    const uint64_t p50 = (N + 1) / 2;
    const uint64_t p75 = (3 * N + 1) / 4;

    // Ascending scan over the entries to locate q1, median, q3.
    uint64_t cumulative = 0;
    long q1 = -1, median = -1, q3 = -1;
    for (int i = 0; i < hist.num_entries; ++i) {
        const long v = static_cast<long>(background_entry_value(hist.entries[i]));
        cumulative += background_entry_count(hist.entries[i]);
        if (q1 < 0 && cumulative >= p25) q1 = v;
        if (median < 0 && cumulative >= p50) median = v;
        if (q3 < 0 && cumulative >= p75) {
            q3 = v;
            break;
        }
    }
    // The cumulative count reaches N, and p75 <= N, so both quartiles are
    // always found for a non-empty histogram.
    if (q1 < 0 || q3 < 0) {
        return result;
    }

    const double iqr = static_cast<double>(q3 - q1);
    const double lower_bound = q1 - iqr_multiplier * iqr;
    const double upper_bound = q3 + iqr_multiplier * iqr;

    // Accumulate inliers.
    uint64_t included_count = 0;
    double weighted_sum = 0.0;
    for (int i = 0; i < hist.num_entries; ++i) {
        const double v = static_cast<double>(background_entry_value(hist.entries[i]));
        if (v < lower_bound || v > upper_bound) continue;
        const uint64_t count = background_entry_count(hist.entries[i]);
        included_count += count;
        weighted_sum += v * static_cast<double>(count);
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
 * special-function dependency on the device. mean tracks the background level,
 * which is small for a photon-counting detector, so the sum is short.
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
 * @brief Robust-Poisson GLM constant background over a sparse histogram.
 *
 * Single-source implementation shared by host and device. Fits a constant
 * Poisson mean with a log link by iteratively reweighted least squares with
 * Huber weighting, reproducing dials::algorithms::RobustPoissonMean over the
 * same per-reflection histogram. The model treats every pixel the same, so the
 * fit only needs the count at each value, which makes the histogram an exact
 * representation.
 *
 * Every pixel value is recorded exactly, so the fit is exact whatever the
 * background level. A full slot table (spill_count > 0) is rejected for the
 * same reason as in tukey_constant_background().
 *
 * The sole numerical divergence from DIALS is the Hessian: DIALS sums H += b
 * per pixel while this uses N * b directly, equal in exact arithmetic but
 * differing in floating-point rounding, so parity holds to 1e-6 rather than
 * bit-for-bit.
 *
 * Paper symbols (constant model, per-pixel term xᵢ = 1): coefficient β, linear
 * predictor η = β, mean μ = exp(η) (log link), link derivative μ′ = dμ/dη,
 * dispersion φ = 1 and variance function v_μ = μ, so √(φ*v_μ) = √μ. The IRLS
 * loop forms the robust score U [Eq 2] and the Fisher information I [Eq 9] and
 * applies the update β <- β + I⁻¹U [Eq 5].
 *
 * Source: dials::algorithms::RobustPoissonMean in
 * dials/algorithms/background/glm/robust_poisson_mean.h.
 */
FFS_HD inline BackgroundResult glm_constant_background(
  const SparseHistogramView &hist) {
    BackgroundResult result;

    // Total pixel count across the histogram.
    uint64_t N = 0;
    for (int i = 0; i < hist.num_entries; ++i) {
        N += background_entry_count(hist.entries[i]);
    }
    // DIALS requires at least min_pixels background pixels to attempt a fit.
    if (N < kGlmMinPixels) {
        return result;
    }
    if (hist.spill_count > 0) {
        return result;
    }

    // Median seed, matching DIALS detail::median (the element at sorted
    // position N/2).
    const uint64_t mid = N / 2;  // 0-based target index
    uint64_t cumulative = 0;
    long median = -1;
    for (int i = 0; i < hist.num_entries; ++i) {
        cumulative += background_entry_count(hist.entries[i]);
        if (cumulative >= mid + 1) {
            median = static_cast<long>(background_entry_value(hist.entries[i]));
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
        // observations). Each entry contributes its count times the per-value
        // term.
        double U = 0.0;
        for (int i = 0; i < hist.num_entries; ++i) {
            const double v =
              static_cast<double>(background_entry_value(hist.entries[i]));
            const uint32_t count = background_entry_count(hist.entries[i]);
            const double res = (v - mu) / svar;  // rᵢ
            const double q = (glm_huber(res, c) - epsi.epsi1) * dmu / svar;
            U += static_cast<double>(count) * q;
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
 * Flattens the aggregator into the shared SparseHistogramView and dispatches to
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
 *         estimate is rejected (no inliers, too few pixels, a full slot table,
 *         or non-convergence), in which case the caller marks the reflection
 *         unintegrated. Mirrors the BackgroundResult::valid channel the GPU
 *         reduction uses.
 */
BackgroundResult compute_background_constant_3d(
  const BackgroundAggregator &data,
  ConstantBackgroundImpl impl = ConstantBackgroundImpl::DialsIndependent,
  BackgroundModel model = BackgroundModel::Constant);
