#ifndef SPOTFIND_H
#define SPOTFIND_H

#include <algorithm>
#include <iostream>
#include <random>

// #include "TinyTIFF/tinytiffwriter.h"

#include <scitbx/array_family/accessors/c_grid.h>
#include <scitbx/array_family/shared.h>
#include "local.h"

namespace af = scitbx::af;

const size_t IMAGE_W = 4000;
const size_t IMAGE_H = 4000;

const af::tiny<int, 2> kernel_size_(3, 3);
const af::tiny<int, 2> image_size_(IMAGE_W, IMAGE_H);

const int min_count_ = 2;
const double threshold_ = 0.0;
const double nsig_b_ = 6.0;
const double nsig_s_ = 3.0;

struct SATData {
    int N;
    double sum;
    double sumsq;

    SATData(int N, double sum, double sumsq) : N(N), sum(sum), sumsq(sumsq) {}
};

/// Class to generate sample image data
template <typename T, typename GAIN_T = double>
class ImageSource {
  public:
    ImageSource()
        : destination_store(IMAGE_W * IMAGE_H),
          image_store(IMAGE_W * IMAGE_H),
          mask_store(IMAGE_W * IMAGE_H),
          gain_store(IMAGE_W * IMAGE_H),
          prefound_store(IMAGE_W * IMAGE_H) {
        // Generate the ref objects we will be using
        src = af::const_ref<T, af::c_grid<2>>(image_store.begin(),
                                              af::c_grid<2>(IMAGE_W, IMAGE_H));
        mask = af::const_ref<bool, af::c_grid<2>>(mask_store.begin(),
                                                  af::c_grid<2>(IMAGE_W, IMAGE_H));
        gain = af::const_ref<GAIN_T, af::c_grid<2>>(gain_store.begin(),
                                                    af::c_grid<2>(IMAGE_W, IMAGE_H));
        dst = af::ref<bool, af::c_grid<2>>(destination_store.begin(),
                                           af::c_grid<2>(IMAGE_W, IMAGE_H));

        // Make an SAT store
        // sat_store.resize(IMAGE_W * IMAGE_H *
        // sizeof(dials::algorithms::DispersionThreshold::Data<double>));

        // Don't mask everything
        std::fill(mask_store.begin(), mask_store.end(), true);
        // Gain of 1
        std::fill(gain_store.begin(), gain_store.end(), 1.0);

        // Create the source image
        // src is a const_ref - we want to access it as a ref
        auto writable_src = af::ref<T, af::c_grid<2>>(image_store.begin(),
                                                      af::c_grid<2>(IMAGE_W, IMAGE_H));

        // Create the image
        std::default_random_engine generator(1);
        std::poisson_distribution<int> poisson(1.0);

        // Write a poisson background over the whole image
        for (int y = 0; y < IMAGE_H; y += 1) {
            for (int x = 0; x < IMAGE_W; x += 1) {
                writable_src(x, y) = poisson(generator);
            }
        }

        int count = 0;
        // Just put some high pixels for now
        for (int y = 0; y < IMAGE_H; y += 42) {
            for (int x = 0; x < IMAGE_W; x += 42) {
                writable_src(x, y) = 100;
                count += 1;
            }
        }

        // for (int y = 0; y < IMAGE_H; y += 1) {
        //   for (int x = 0; x < IMAGE_W; x += 1) {
        //     writable_src(y,x) = 1;
        //   }
        // }

        // int n = 0;
        // long sum = 0;
        // long sumsq = 0;
        // for (int y = 0; y < IMAGE_H; ++y) {
        //   for (int x = 0; x < IMAGE_W; ++x) {
        //     n += 1;
        //     sum += src(y,x);
        //     sumsq += src(y,x)*src(y,x);
        //   }
        // }
        // std::cout << "Diagnostics:\n  n:    " << n << "\n  sum:   " << sum
        //           << "\n  sumsq: " << sumsq << std::endl;

        // Run spotfinding to have a "precalculated" result to validate against
        auto prefdst = af::ref<bool, af::c_grid<2>>(prefound_store.begin(),
                                                    af::c_grid<2>(IMAGE_W, IMAGE_H));
        auto algo = dials::algorithms::DispersionThreshold(
          image_size_, kernel_size_, nsig_b_, nsig_s_, threshold_, min_count_);
        // Convert internal gain to double for the function
        af::shared<double> _pref_gain_store(IMAGE_W * IMAGE_H);
        std::fill(_pref_gain_store.begin(), _pref_gain_store.end(), 1.0);
        auto _pref_gain = af::const_ref<double, af::c_grid<2>>(
          _pref_gain_store.begin(), af::c_grid<2>(IMAGE_W, IMAGE_H));

        algo.threshold_w_gain(src, mask, _pref_gain, prefdst);

        // Save the SAT table for diagnostics
        prefound_SAT.reserve(IMAGE_W * IMAGE_H);
        dials::algorithms::DispersionThreshold::Data<T>* sat =
          reinterpret_cast<dials::algorithms::DispersionThreshold::Data<T>*>(
            &algo.buffer_.front());
        for (int i = 0; i < (IMAGE_W * IMAGE_H); ++i) {
            prefound_SAT.emplace_back(SATData(sat[i].m, sat[i].x, sat[i].y));
        }

        write_array("dispersion.tif", prefdst);
#ifdef BENCHMARK
        // If doing a benchmark, make sure to avoid inlining issues
        benchmark::ClobberMemory();
#endif
    }

    bool validate_dst(af::const_ref<bool, af::c_grid<2>> tocheck) {
        auto prefdst = af::ref<bool, af::c_grid<2>>(prefound_store.begin(),
                                                    af::c_grid<2>(IMAGE_W, IMAGE_H));
        if ((tocheck.accessor()[0] != prefdst.accessor()[0])
            && (tocheck.accessor()[1] != prefdst.accessor()[1])) {
            std::cout << "Validation Failed: Size mismatch" << std::endl;
            return false;
        }
        int fail_count = 0;
        for (int y = 0; y < IMAGE_H; y += 1) {
            for (int x = 0; x < IMAGE_W; x += 1) {
                if (tocheck(y, x) != prefdst(y, x)) {
                    if (fail_count < 5) {
                        std::cout << "Validation Failed: " << x << ", " << y << " "
                                  << tocheck(y, x) << " instead of " << prefdst(y, x)
                                  << std::endl;
                    }
                    fail_count += 1;
                    if (fail_count == 5) {
                        std::cout << "..." << std::endl;
                    }
                }
            }
        }
        if (fail_count) {
            std::cout << "Total mismatches: " << fail_count << std::endl;
        }
        return fail_count == 0;
    }

    /// Kill the dst explicitly
    void reset_dst() {
        std::fill(destination_store.begin(), destination_store.end(), false);
    }

    // // Write an array_family object to a TIFF file
    template <typename IMSRC>
    void write_array(const char* filename, IMSRC image) {
        //     // benchmark::ClobberMemory();
        //     TinyTIFFFile* tif = TinyTIFFWriter_open(filename, 16, IMAGE_W, IMAGE_H);
        //     std::vector<uint16_t> img;
        //     img.reserve(image.accessor()[0] * image.accessor()[1]);
        //     for (int y = 0; y < IMAGE_H; ++y) {
        //         for (int x = 0; x < IMAGE_W; ++x) {
        //             img.push_back(image(y, x));
        //         }
        //     }
        //     TinyTIFFWriter_writeImage(tif, &img.front());
        //     TinyTIFFWriter_close(tif);
    }

    af::shared<bool> destination_store;
    af::shared<T> image_store;
    af::shared<bool> mask_store;
    af::shared<GAIN_T> gain_store;
    af::shared<bool> prefound_store;  // For checking results
    std::vector<SATData> prefound_SAT;
    // std::vector<unsigned char> sat_store;

    af::const_ref<T, af::c_grid<2>> src;
    af::const_ref<bool, af::c_grid<2>> mask;
    af::const_ref<GAIN_T, af::c_grid<2>> gain;
    af::ref<bool, af::c_grid<2>> dst;
};

#endif
