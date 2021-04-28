#include <benchmark/benchmark.h>

#include <iostream>

#include "baseline.h"
#include "miniapp.h"
#include "spotfind_test_utils.h"

using dials::algorithms::DispersionExtendedThreshold;
using dials::algorithms::DispersionThreshold;

template <class T>
static void BM_standard_dispersion(benchmark::State& state) {
    ImageSource<T> src;
    // BeginTask task("dials.dispersion.benchmark", "dispersion");
    auto algo = DispersionThreshold(
      image_size_, kernel_size_, nsig_b_, nsig_s_, threshold_, min_count_);

    for (auto _ : state) {
        algo.threshold(src.src, src.mask, src.dst);
    }
}
BENCHMARK_TEMPLATE(BM_standard_dispersion, double)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_standard_dispersion, float)->Unit(benchmark::kMillisecond);

// template <class T>
static void BM_C_API_dispersion(benchmark::State& state) {
    ImageSource<uint32_t> src;

    auto finder = create_spotfinder(IMAGE_W, IMAGE_H);

    // Convert src to an image_t
    image_t image;
    image.fast = IMAGE_W;
    image.slow = IMAGE_H;
    image.mask = new uint8_t[IMAGE_H * IMAGE_W];
    image.data = new image_t_type[IMAGE_W * IMAGE_H];
    // Copy the data
    for (int i = 0; i < IMAGE_W * IMAGE_H; ++i) {
        image.mask[i] = src.mask[i];
        image.data[i] = src.src[i];
    }

    for (auto _ : state) {
        spotfinder_standard_dispersion(finder, &image);
    }
    free_spotfinder(finder);
    delete[] image.mask;
    delete[] image.data;
}
BENCHMARK(BM_C_API_dispersion)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
