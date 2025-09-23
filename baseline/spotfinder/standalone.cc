
#include "standalone.h"

#include <h5read.h>

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

using std::span;

const std::array<int, 2> kernel_size_{3, 3};
const int min_count_ = 2;
const double threshold_ = 0.0;
const double nsig_b_ = 6.0;
const double nsig_s_ = 3.0;

namespace no_tbx {

/**
 * A class to compute the threshold using index of dispersion
 */
template <typename T>
class DispersionThreshold {
  public:
    /**
     * Enable more efficient memory usage by putting components required for the
     * summed area table closer together in memory
     */
    struct Data {
        int m;
        T x;
        T y;
    };

    DispersionThreshold(std::array<int, 2> image_size,
                        std::array<int, 2> kernel_size,
                        double nsig_b,
                        double nsig_s,
                        double threshold,
                        int min_count)
        : image_size_(image_size),
          kernel_size_(kernel_size),
          nsig_b_(nsig_b),
          nsig_s_(nsig_s),
          threshold_(threshold),
          min_count_(min_count) {
        // Check the input
        assert(threshold_ >= 0);
        assert(nsig_b >= 0 && nsig_s >= 0);
        assert(image_size[0] > 0 && image_size[1] > 0);
        assert(kernel_size[0] > 0 && kernel_size[1] > 0);

        // Ensure the min counts are valid
        std::size_t num_kernel = (2 * kernel_size[0] + 1) * (2 * kernel_size[1] + 1);
        if (min_count_ <= 0) {
            min_count_ = num_kernel;
        } else {
            assert(min_count_ <= num_kernel && min_count_ > 1);
        }

        table_.resize(image_size[0] * image_size[1]);
    }

    /**
     * Compute the summed area tables for the mask, src and src^2.
     * @param src The input array
     * @param mask The mask array
     */
    void compute_sat(span<Data> table,
                     const span<const T> src,
                     const span<const bool> mask) {
        // Largest value to consider
        const T BIG = (1 << 24);  // About 16m counts

        // Get the size of the image

        auto [ysize, xsize] = image_size_;

        // Create the summed area table
        for (std::size_t j = 0, k = 0; j < ysize; ++j) {
            int m = 0;
            T x = 0;
            T y = 0;
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int mm = (mask[k] && src[k] < BIG) ? 1 : 0;
                m += mm;
                x += mm * src[k];
                y += mm * src[k] * src[k];
                if (j == 0) {
                    table[k].m = m;
                    table[k].x = x;
                    table[k].y = y;
                } else {
                    table[k].m = table[k - xsize].m + m;
                    table[k].x = table[k - xsize].x + x;
                    table[k].y = table[k - xsize].y + y;
                }
            }
        }
    }

    /**
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param dst The output array
     */
    void compute_threshold(span<const Data> table,
                           const span<const T> src,
                           const span<const bool> mask,
                           span<bool> dst) {
        // Get the size of the image
        auto [ysize, xsize] = image_size_;

        // The kernel size
        int kxsize = kernel_size_[1];
        int kysize = kernel_size_[0];

        // Calculate the local mean at every point
        for (std::size_t j = 0, k = 0; j < ysize; ++j) {
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int i0 = i - kxsize - 1, i1 = i + kxsize;
                int j0 = j - kysize - 1, j1 = j + kysize;
                i1 = i1 < xsize ? i1 : xsize - 1;
                j1 = j1 < ysize ? j1 : ysize - 1;
                int k0 = j0 * xsize;
                int k1 = j1 * xsize;

                // Compute the number of points valid in the local area,
                // the sum of the pixel values and the sum of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                double y = 0;
                if (i0 >= 0 && j0 >= 0) {
                    const Data &d00 = table[k0 + i0];
                    const Data &d10 = table[k1 + i0];
                    const Data &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= 0) {
                    const Data &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= 0) {
                    const Data &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;
                y += d11.y;

                // Compute the thresholds
                dst[k] = false;
                if (mask[k] && m >= min_count_ && x >= 0 && src[k] > threshold_) {
                    double a = m * y - x * x - x * (m - 1);
                    double b = m * src[k] - x;
                    double c = x * nsig_b_ * std::sqrt(2 * (m - 1));
                    double d = nsig_s_ * std::sqrt(x * m);
                    dst[k] = a > c && b > d;
                }
            }
        }
    }

    /**
     * Compute the threshold for the given image and mask.
     * @param src - The input image array.
     * @param mask - The mask array.
     * @param dst - The destination array.
     */
    void threshold(const span<const T> src,
                   const span<const bool> mask,
                   span<bool> dst) {
        // check the input
        assert(src.size() >= image_size_[0] * image_size_[1]);
        assert(src.size() == mask.size());
        assert(src.size() == dst.size());

        // compute the summed area table
        compute_sat(table_, src, mask);

        // Compute the image threshold
        auto table_span = span<Data>{table_.data(), table_.size()};
        compute_threshold(table_span, src, mask, dst);
    }

  private:
    std::array<int, 2> image_size_;
    std::array<int, 2> kernel_size_;
    double nsig_b_;
    double nsig_s_;
    double threshold_;
    int min_count_;
    std::vector<Data> table_;
};

}  // namespace no_tbx

template class StandaloneSpotfinder<float>;
template class StandaloneSpotfinder<double>;

template <typename T>
void StandaloneSpotfinder<T>::StandaloneSpotfinderImplDeleter::operator()(
  StandaloneSpotfinderImpl *ptr) const {
    delete ptr;
}

template <typename T>
class StandaloneSpotfinder<T>::StandaloneSpotfinderImpl {
  public:
    StandaloneSpotfinderImpl(size_t width, size_t height)
        : width(width),
          height(height),
          results(width * height),
          algorithm({static_cast<int>(height), static_cast<int>(width)},
                    kernel_size_,
                    nsig_b_,
                    nsig_s_,
                    threshold_,
                    min_count_) {}

    size_t width;
    size_t height;
    std::vector<uint8_t> results;
    no_tbx::DispersionThreshold<T> algorithm;
};

template <typename T>
StandaloneSpotfinder<T>::StandaloneSpotfinder(size_t width, size_t height) {
    // Can't use make_unique with custom deleter
    auto obj = new StandaloneSpotfinderImpl(width, height);
    impl =
      std::unique_ptr<StandaloneSpotfinderImpl, StandaloneSpotfinderImplDeleter>(obj);
}

template <typename T>
auto StandaloneSpotfinder<T>::standard_dispersion(const span<const T> image,
                                                  const span<const bool> mask)
  -> span<const bool> {
    auto results =
      span<bool>{reinterpret_cast<bool *>(impl->results.data()), impl->results.size()};

    impl->algorithm.threshold(image, mask, results);

    return results;
}
template <typename T>
auto StandaloneSpotfinder<T>::standard_dispersion(const span<const T> image,
                                                  const span<const uint8_t> mask)
  -> span<const bool> {
    auto results =
      span<bool>{reinterpret_cast<bool *>(impl->results.data()), impl->results.size()};

    auto c_mask =
      span<const bool>{reinterpret_cast<const bool *>(mask.data()), mask.size()};
    impl->algorithm.threshold(image, c_mask, results);

    return results;
}
