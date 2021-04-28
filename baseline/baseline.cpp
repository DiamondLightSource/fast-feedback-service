
#include "baseline.h"

#include <dials/algorithms/image/filter/distance.h>
#include <dials/algorithms/image/filter/index_of_dispersion_filter.h>
#include <dials/algorithms/image/filter/mean_and_variance.h>
#include <dials/error.h>
#include <scitbx/array_family/ref_reductions.h>
#include <scitbx/array_family/tiny_types.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "spotfind_test_utils.h"

namespace baseline {

// using namespace dials;
using namespace dials::algorithms;

/**
 * A class to compute the threshold using index of dispersion
 */
class DispersionThreshold {
  public:
    /**
     * Enable more efficient memory usage by putting components required for the
     * summed area table closer together in memory
     */
    template <typename T>
    struct Data {
        int m;
        T x;
        T y;
    };

    DispersionThreshold(int2 image_size,
                        int2 kernel_size,
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
        DIALS_ASSERT(threshold_ >= 0);
        DIALS_ASSERT(nsig_b >= 0 && nsig_s >= 0);
        DIALS_ASSERT(image_size.all_gt(0));
        DIALS_ASSERT(kernel_size.all_gt(0));

        // Ensure the min counts are valid
        std::size_t num_kernel = (2 * kernel_size[0] + 1) * (2 * kernel_size[1] + 1);
        if (min_count_ <= 0) {
            min_count_ = num_kernel;
        } else {
            DIALS_ASSERT(min_count_ <= num_kernel && min_count_ > 1);
        }

        // Allocate the buffer
        std::size_t element_size = sizeof(Data<double>);
        buffer_.resize(element_size * image_size[0] * image_size[1]);
    }

    /**
     * Compute the summed area tables for the mask, src and src^2.
     * @param src The input array
     * @param mask The mask array
     */
    template <typename T>
    void compute_sat(af::ref<Data<T>> table,
                     const af::const_ref<T, af::c_grid<2>> &src,
                     const af::const_ref<bool, af::c_grid<2>> &mask) {
        // Largest value to consider
        const T BIG = (1 << 24);  // About 16m counts

        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

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
    template <typename T>
    void compute_threshold(af::ref<Data<T>> table,
                           const af::const_ref<T, af::c_grid<2>> &src,
                           const af::const_ref<bool, af::c_grid<2>> &mask,
                           af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

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
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data<T> &d11 = table[k1 + i1];
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
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param gain - The gain array
     * @param dst The output array
     */
    template <typename T>
    void compute_threshold(af::ref<Data<T>> table,
                           const af::const_ref<T, af::c_grid<2>> &src,
                           const af::const_ref<bool, af::c_grid<2>> &mask,
                           const af::const_ref<double, af::c_grid<2>> &gain,
                           af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

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
                // the sum of the pixel values and the num of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                double y = 0;
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;
                y += d11.y;

                // Compute the thresholds
                dst[k] = false;
                if (mask[k] && m >= min_count_ && x >= 0 && src[k] > threshold_) {
                    double a = m * y - x * x;
                    double b = m * src[k] - x;
                    double c = gain[k] * x * (m - 1 + nsig_b_ * std::sqrt(2 * (m - 1)));
                    double d = nsig_s_ * std::sqrt(gain[k] * x * m);
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
    template <typename T>
    void threshold(const af::const_ref<T, af::c_grid<2>> &src,
                   const af::const_ref<bool, af::c_grid<2>> &mask,
                   af::ref<bool, af::c_grid<2>> dst) {
        // check the input
        DIALS_ASSERT(src.accessor().all_eq(image_size_));
        DIALS_ASSERT(src.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(dst.accessor()));

        // Get the table
        DIALS_ASSERT(sizeof(T) <= sizeof(double));

        // Cast the buffer to the table type
        af::ref<Data<T>> table(reinterpret_cast<Data<T> *>(&buffer_[0]),
                               buffer_.size());

        // compute the summed area table
        compute_sat(table, src, mask);

        // Compute the image threshold
        compute_threshold(table, src, mask, dst);
    }

    /**
     * Compute the threshold for the given image and mask.
     * @param src - The input image array.
     * @param mask - The mask array.
     * @param gain - The gain array
     * @param dst - The destination array.
     */
    template <typename T>
    void threshold_w_gain(const af::const_ref<T, af::c_grid<2>> &src,
                          const af::const_ref<bool, af::c_grid<2>> &mask,
                          const af::const_ref<double, af::c_grid<2>> &gain,
                          af::ref<bool, af::c_grid<2>> dst) {
        // check the input
        DIALS_ASSERT(src.accessor().all_eq(image_size_));
        DIALS_ASSERT(src.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(gain.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(dst.accessor()));

        // Get the table
        DIALS_ASSERT(sizeof(T) <= sizeof(double));

        // Cast the buffer to the table type
        af::ref<Data<T>> table((Data<T> *)&buffer_[0], buffer_.size());

        // compute the summed area table
        compute_sat(table, src, mask);

        // Compute the image threshold
        compute_threshold(table, src, mask, gain, dst);
    }

  private:
    int2 image_size_;
    int2 kernel_size_;
    double nsig_b_;
    double nsig_s_;
    double threshold_;
    int min_count_;
    std::vector<char> buffer_;
};

/**
 * A class to compute the threshold using index of dispersion
 */
class DispersionExtendedThreshold {
  public:
    /**
     * Enable more efficient memory usage by putting components required for the
     * summed area table closer together in memory
     */
    template <typename T>
    struct Data {
        int m;
        T x;
        T y;
    };

    DispersionExtendedThreshold(int2 image_size,
                                int2 kernel_size,
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
        DIALS_ASSERT(threshold_ >= 0);
        DIALS_ASSERT(nsig_b >= 0 && nsig_s >= 0);
        DIALS_ASSERT(image_size.all_gt(0));
        DIALS_ASSERT(kernel_size.all_gt(0));

        // Ensure the min counts are valid
        std::size_t num_kernel = (2 * kernel_size[0] + 1) * (2 * kernel_size[1] + 1);
        if (min_count_ <= 0) {
            min_count_ = num_kernel;
        } else {
            DIALS_ASSERT(min_count_ <= num_kernel && min_count_ > 1);
        }

        // Allocate the buffer
        std::size_t element_size = sizeof(Data<double>);
        buffer_.resize(element_size * image_size[0] * image_size[1]);
    }

    /**
     * Compute the summed area tables for the mask, src and src^2.
     * @param src The input array
     * @param mask The mask array
     */
    template <typename T>
    void compute_sat(af::ref<Data<T>> table,
                     const af::const_ref<T, af::c_grid<2>> &src,
                     const af::const_ref<bool, af::c_grid<2>> &mask) {
        // Largest value to consider
        const T BIG = (1 << 24);  // About 16m counts

        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

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
    template <typename T>
    void compute_dispersion_threshold(af::ref<Data<T>> table,
                                      const af::const_ref<T, af::c_grid<2>> &src,
                                      const af::const_ref<bool, af::c_grid<2>> &mask,
                                      af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

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
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;
                y += d11.y;

                // Compute the thresholds
                dst[k] = false;
                if (mask[k] && m >= min_count_ && x >= 0) {
                    double a = m * y - x * x - x * (m - 1);
                    double c = x * nsig_b_ * std::sqrt(2 * (m - 1));
                    dst[k] = (a > c);
                }
            }
        }
    }

    /**
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param gain - The gain array
     * @param dst The output array
     */
    template <typename T>
    void compute_dispersion_threshold(af::ref<Data<T>> table,
                                      const af::const_ref<T, af::c_grid<2>> &src,
                                      const af::const_ref<bool, af::c_grid<2>> &mask,
                                      const af::const_ref<double, af::c_grid<2>> &gain,
                                      af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

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
                // the sum of the pixel values and the num of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                double y = 0;
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;
                y += d11.y;

                // Compute the thresholds
                dst[k] = false;
                if (mask[k] && m >= min_count_ && x >= 0) {
                    double a = m * y - x * x;
                    double c = gain[k] * x * (m - 1 + nsig_b_ * std::sqrt(2 * (m - 1)));
                    dst[k] = (a > c);
                }
            }
        }
    }

    /**
     * Erode the dispersion mask
     * @param dst The dispersion mask
     */
    void erode_dispersion_mask(const af::const_ref<bool, af::c_grid<2>> &mask,
                               af::ref<bool, af::c_grid<2>> dst) {
        // The distance array
        af::versa<int, af::c_grid<2>> distance(dst.accessor(), 0);

        // Compute the chebyshev distance to the nearest valid background pixel
        chebyshev_distance(dst, false, distance.ref());

        // The erosion distance
        std::size_t erosion_distance = std::min(kernel_size_[0], kernel_size_[1]);

        // Compute the eroded mask
        for (std::size_t k = 0; k < dst.size(); ++k) {
            if (mask[k]) {
                dst[k] = !(dst[k] && distance[k] >= erosion_distance);
            } else {
                dst[k] = false;
            }
        }
    }

    /**
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param dst The output array
     */
    template <typename T>
    void compute_final_threshold(af::ref<Data<T>> table,
                                 const af::const_ref<T, af::c_grid<2>> &src,
                                 const af::const_ref<bool, af::c_grid<2>> &mask,
                                 af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

        // The kernel size
        int kxsize = kernel_size_[1] + 2;
        int kysize = kernel_size_[0] + 2;

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
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;

                // Compute the thresholds. The pixel is marked True if:
                // 1. The pixel is valid
                // 2. It has 1 or more unmasked neighbours
                // 3. It is within the dispersion masked region
                // 4. It is greater than the global threshold
                // 5. It is greater than the local mean threshold
                //
                // Otherwise it is false
                if (mask[k] && m >= 0 && x >= 0) {
                    bool dispersion_mask = !dst[k];
                    bool global_mask = src[k] > threshold_;
                    double mean = (m >= 2 ? (x / m) : 0);
                    bool local_mask = src[k] >= (mean + nsig_s_ * std::sqrt(mean));
                    dst[k] = dispersion_mask && global_mask && local_mask;
                } else {
                    dst[k] = false;
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
    template <typename T>
    void compute_final_threshold(af::ref<Data<T>> table,
                                 const af::const_ref<T, af::c_grid<2>> &src,
                                 const af::const_ref<bool, af::c_grid<2>> &mask,
                                 const af::const_ref<double, af::c_grid<2>> &gain,
                                 af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

        // The kernel size
        int kxsize = kernel_size_[1] + 2;
        int kysize = kernel_size_[0] + 2;

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
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;

                // Compute the thresholds. The pixel is marked True if:
                // 1. The pixel is valid
                // 2. It has 1 or more unmasked neighbours
                // 3. It is within the dispersion masked region
                // 4. It is greater than the global threshold
                // 5. It is greater than the local mean threshold
                //
                // Otherwise it is false
                if (mask[k] && m >= 0 && x >= 0) {
                    bool dispersion_mask = !dst[k];
                    bool global_mask = src[k] > threshold_;
                    double mean = (m >= 2 ? (x / m) : 0);
                    bool local_mask =
                      src[k] >= (mean + nsig_s_ * std::sqrt(gain[k] * mean));
                    dst[k] = dispersion_mask && global_mask && local_mask;
                } else {
                    dst[k] = false;
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
    template <typename T>
    void threshold(const af::const_ref<T, af::c_grid<2>> &src,
                   const af::const_ref<bool, af::c_grid<2>> &mask,
                   af::ref<bool, af::c_grid<2>> dst) {
        // check the input
        DIALS_ASSERT(src.accessor().all_eq(image_size_));
        DIALS_ASSERT(src.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(dst.accessor()));

        // Get the table
        DIALS_ASSERT(sizeof(T) <= sizeof(double));

        // Cast the buffer to the table type
        af::ref<Data<T>> table(reinterpret_cast<Data<T> *>(&buffer_[0]),
                               buffer_.size());

        // compute the summed area table
        compute_sat(table, src, mask);

        // Compute the dispersion threshold. This output is in dst which contains
        // a mask where 1 is valid background and 0 is invalid pixels and stuff
        // above the dispersion threshold
        compute_dispersion_threshold(table, src, mask, dst);

        // Erode the dispersion mask
        erode_dispersion_mask(mask, dst);

        // Compute the summed area table again now excluding the threshold pixels
        compute_sat(table, src, dst);

        // Compute the final threshold
        compute_final_threshold(table, src, mask, dst);
    }

    /**
     * Compute the threshold for the given image and mask.
     * @param src - The input image array.
     * @param mask - The mask array.
     * @param gain - The gain array
     * @param dst - The destination array.
     */
    template <typename T>
    void threshold_w_gain(const af::const_ref<T, af::c_grid<2>> &src,
                          const af::const_ref<bool, af::c_grid<2>> &mask,
                          const af::const_ref<double, af::c_grid<2>> &gain,
                          af::ref<bool, af::c_grid<2>> dst) {
        // check the input
        DIALS_ASSERT(src.accessor().all_eq(image_size_));
        DIALS_ASSERT(src.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(gain.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(dst.accessor()));

        // Get the table
        DIALS_ASSERT(sizeof(T) <= sizeof(double));

        // Cast the buffer to the table type
        af::ref<Data<T>> table((Data<T> *)&buffer_[0], buffer_.size());

        // compute the summed area table
        compute_sat(table, src, mask);

        // Compute the dispersion threshold. This output is in dst which contains
        // a mask where 1 is valid background and 0 is invalid pixels and stuff
        // above the dispersion threshold
        compute_dispersion_threshold(table, src, mask, gain, dst);

        // Erode the dispersion mask
        erode_dispersion_mask(mask, dst);

        // Compute the summed area table again now excluding the threshold pixels
        compute_sat(table, src, dst);

        // Compute the final threshold
        compute_final_threshold(table, src, mask, gain, dst);
    }

  private:
    int2 image_size_;
    int2 kernel_size_;
    double nsig_b_;
    double nsig_s_;
    double threshold_;
    int min_count_;
    std::vector<char> buffer_;
};

}  // namespace baseline

template <typename T, typename internal_T = T>
class _spotfind_context {
  public:
    af::ref<bool, af::c_grid<2>> dst;
    bool *_dest_store = nullptr;
    af::tiny<int, 2> size;

    af::ref<internal_T, af::c_grid<2>> src_converted;
    internal_T *_src_converted_store;

    baseline::DispersionThreshold algo;

    _spotfind_context(size_t width, size_t height)
        : size(width, height),
          algo(size, kernel_size_, nsig_b_, nsig_s_, threshold_, min_count_) {
        _dest_store = new bool[width * height];
        dst =
          af::ref<bool, af::c_grid<2>>(_dest_store, af::c_grid<2>(IMAGE_W, IMAGE_H));
        // Make a place to convert sources to the internal type
        _src_converted_store = new internal_T[width * height];
        src_converted = af::ref<internal_T, af::c_grid<2>>(
          _src_converted_store, af::c_grid<2>(IMAGE_W, IMAGE_H));
    }
    ~_spotfind_context() {
        delete[] _dest_store;
        delete[] _src_converted_store;
    }
    void threshold(const af::const_ref<internal_T, af::c_grid<2>> &src,
                   const af::const_ref<bool, af::c_grid<2>> &mask) {
        algo.threshold(src_converted, mask, dst);
    }
};

void *create_spotfinder(size_t width, size_t height) {
    return new _spotfind_context<image_t_type, double>(width, height);
}
void free_spotfinder(void *context) {
    delete reinterpret_cast<_spotfind_context<image_t_type, double> *>(context);
}

uint32_t spotfinder_standard_dispersion(void *context, image_t *image) {
    auto ctx = reinterpret_cast<_spotfind_context<image_t_type, double> *>(context);

    // mask needs to convert uint8_t to bool
    auto mask = af::const_ref<bool, af::c_grid<2>>(
      reinterpret_cast<bool *>(image->mask), af::c_grid<2>(IMAGE_W, IMAGE_H));

    // Convert all items from the source image to
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        ctx->src_converted[i] = image->data[i];
    }

    ctx->threshold(ctx->src_converted, mask);

    // Let's count the number of destination pixels for now
    uint32_t pixel_count = 0;
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        pixel_count += ctx->dst[i];
    }
    return pixel_count;
}
