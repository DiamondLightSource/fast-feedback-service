#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

static constexpr std::size_t VECTOR_LIMIT = 64;

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
    // Use two data structures for the histogram
    // - a small array for small counts (< VECTOR_LIMIT) (vast majority of pixels, efficient for adding a large number of low-value pixels)
    // - an unordered map pointer for large counts (sparse, infrequent, efficient for adding a low number of high-value pixels (perhaps outliers))
    std::array<std::size_t, VECTOR_LIMIT> _small_hist{};
    std::unordered_map<int, std::size_t> *_large_hist = nullptr;
    int n_pixels = 0;
};

// Simple background with tukey outlier for now.
std::tuple<double, double> compute_background_constant_3d(
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
