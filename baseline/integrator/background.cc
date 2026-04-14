#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

constexpr std::size_t VECTOR_LIMIT = 64;

class BackgroundAggregator {
  public:
    BackgroundAggregator() {}
    void add(int x) {
        if (x >= 0 && x < VECTOR_LIMIT) {
            ++_small_hist[x];
        } else {
            ++_large_hist[x];
        }
        ++n_pixels;
    }
    int num_pixels() const {
        return n_pixels;
    }
    std::vector<std::size_t> small_hist() const {
        return _small_hist;
    }
    std::unordered_map<int, std::size_t> large_hist() const {
        return _large_hist;
    }

  private:
    std::vector<std::size_t> _small_hist = std::vector<std::size_t>(VECTOR_LIMIT, 0);
    std::unordered_map<int, std::size_t> _large_hist;
    int n_pixels = 0;
};

// Compute constant 3d glm background model
// int min_pixels=10

// Actually do simple with tukey outlier

std::tuple<double, double> compute_background_constant_3d(BackgroundAggregator data) {
    // first do tukey outlier rejection based on 1.5 IQR multiplier.
    double iqr_multiplier = 1.5;
    int N = data.num_pixels();

    std::size_t p25 = (N + 3) / 4;
    std::size_t p50 = (N + 1) / 2;
    std::size_t p75 = (3 * N + 1) / 4;

    std::vector<std::size_t> small_hist = data.small_hist();
    std::unordered_map<int, std::size_t> large_hist = data.large_hist();

    // iterate the vector.
    std::size_t cumulative = 0;
    int q1 = -1, median = -1, q3 = -1;
    for (int value = 0; value < small_hist.size(); ++value) {
        cumulative += small_hist[value];

        if (q1 < 0 && cumulative >= p25) q1 = value;
        if (median < 0 && cumulative >= p50) median = value;
        if (q3 < 0 && cumulative >= p75) {
            q3 = value;
            break;  // may be able to stop early
        }
    }
    // iterate the hist if not all found.
    if (q3 < 0) {
        std::vector<int> keys;
        keys.reserve(large_hist.size());
        for (auto &[k, _] : large_hist) keys.push_back(k);

        std::sort(keys.begin(), keys.end());

        for (int value : keys) {
            cumulative += large_hist[value];

            if (q1 < 0 && cumulative >= p25) q1 = value;
            if (median < 0 && cumulative >= p50) median = value;
            if (q3 < 0 && cumulative >= p75) {
                q3 = value;
                break;
            }
        }
    }
    int iqr = q3 - q1;
    double upper_bound = q3 + (iqr * iqr_multiplier);
    double lower_bound = q1 - (iqr * iqr_multiplier);

    std::size_t included_count = 0;
    double weighted_sum = 0.0;

    // ---- Small values (vector) ----
    for (std::size_t value = 0; value < small_hist.size(); ++value) {
        if (value < lower_bound || value > upper_bound) continue;

        std::size_t count = small_hist[value];
        included_count += count;
        weighted_sum += value * static_cast<double>(count);
    }

    // ---- Large outliers (unordered_map) ----
    for (const auto &[value, count] : large_hist) {
        if (value < lower_bound || value > upper_bound) continue;

        included_count += count;
        weighted_sum += value * static_cast<double>(count);
    }

    // Avoid division by zero if everything was excluded
    if (included_count == 0)
        throw std::runtime_error("No counts included in background calculation");

    return std::make_tuple(weighted_sum / static_cast<double>(included_count),
                           weighted_sum);
}