/**
 * @file background.hpp
 * @brief Background estimation for baseline CPU integration.
 */

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <unordered_map>

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
 * @return {mean background, weighted sum of included pixel values}
 */
std::tuple<double, double> compute_background_constant_3d(
  const BackgroundAggregator &data);
