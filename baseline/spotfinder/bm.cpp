#include <benchmark/benchmark.h>

#include <cassert>
#include <iostream>
#include <span>
#include <type_traits>
#include <vector>

#include "baseline.h"
#include "h5read.h"
#include "spotfind_test_utils.h"
#include "standalone.h"

using std::cout;
using std::endl;

#ifdef HAVE_DIALS
using dials::algorithms::DispersionExtendedThreshold;
using dials::algorithms::DispersionThreshold;

namespace af = scitbx::af;
#endif
template <typename T = H5Read::image_type>
class ImageSource {
  public:
    H5Read reader;

    ImageSource(const int sample_image_number = 5) {
        size_t num_pixels = reader.get_image_fast() * reader.get_image_slow();
        auto data_store = std::vector<H5Read::image_type>(num_pixels);
        reader.get_image_into(sample_image_number, data_store.data());
        // Convert to our internal store type
        _source = std::vector<T>(data_store.begin(), data_store.end());
        _result = std::vector<uint8_t>(num_pixels);
    }

    auto image_data() const -> const std::span<const T> {
        return {_source.data(), _source.size()};
    }

    auto mask_data() const -> const std::span<const bool> {
        static_assert(sizeof(uint8_t) == sizeof(bool));
        return {reinterpret_cast<const bool*>(reader.get_mask().value().data()),
                reader.get_image_fast() * reader.get_image_slow()};
    };
    auto result_buffer() -> std::span<uint8_t> {
        return {_result.data(), slow() * fast()};
    }

    auto fast() const -> size_t {
        return reader.get_image_fast();
    }
    auto slow() const -> size_t {
        return reader.get_image_slow();
    }

    /// Return an H5Read image object for direct passing to C API
    auto h5read_image() const -> image_t {
        static_assert(std::is_same<T, H5Read::image_type>::value,
                      "Cannot convert non-uint16_t buffers to image_t");
        return {
          .data = const_cast<H5Read::image_type*>(image_data().data()),
          .mask = reinterpret_cast<uint8_t*>(const_cast<bool*>(mask_data().data())),
          .slow = slow(),
          .fast = fast(),
        };
    }

#ifdef HAVE_DIALS
    auto image_data_ref() const -> af::const_ref<T, af::c_grid<2>> {
        return af::const_ref<T, af::c_grid<2>>(_source.data(),
                                               af::c_grid<2>(fast(), slow()));
    }
    auto mask_data_ref() const -> af::const_ref<bool, af::c_grid<2>> {
        return {const_cast<bool*>(mask_data().data()), af::c_grid<2>(fast(), slow())};
    }
    auto result_buffer_ref() -> af::ref<bool, af::c_grid<2>> {
        return {reinterpret_cast<bool*>(_result.data()), af::c_grid<2>(fast(), slow())};
    };
#endif
  private:
    std::vector<T> _source;
    std::vector<uint8_t> _result;
};

#ifdef HAVE_DIALS
template <class T>
static void BM_standard_dispersion(benchmark::State& state) {
    ImageSource<T> src;
    // BeginTask task("dials.dispersion.benchmark", "dispersion");
    auto algo =
      DispersionThreshold({static_cast<int>(src.fast()), static_cast<int>(src.slow())},
                          kernel_size_,
                          nsig_b_,
                          nsig_s_,
                          threshold_,
                          min_count_);

    auto data_ref = src.image_data_ref();
    auto mask_ref = src.mask_data_ref();
    auto result_ref = src.result_buffer_ref();

    for (auto _ : state) {
        algo.threshold<T>(data_ref, mask_ref, result_ref);
    }
    // Double check this against pre-calculated
    // assert(src.validate_dst(src.dst));
    // Count the number of spots in dst
    uint32_t pixel_count = 0;
    for (int i = 0; i < (IMAGE_H * IMAGE_W); ++i) {
        pixel_count += src.result_buffer()[i];
    }
}
BENCHMARK_TEMPLATE(BM_standard_dispersion, double)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_standard_dispersion, float)->Unit(benchmark::kMillisecond);

static void BM_C_API_dispersion(benchmark::State& state) {
    ImageSource<H5Read::image_type> src;
    auto image = src.h5read_image();

    auto finder = spotfinder_create(src.fast(), src.slow());

    uint32_t spots = 0;
    for (auto _ : state) {
        spots = spotfinder_standard_dispersion(finder, &image);
    }
    spotfinder_free(finder);
}
BENCHMARK(BM_C_API_dispersion)->Unit(benchmark::kMillisecond);

#endif

static void BM_Standalone_dispersion_w_convert(benchmark::State& state) {
    ImageSource<uint16_t> src;
    auto image = src.h5read_image();
    auto finder = StandaloneSpotfinder<double>(src.fast(), src.slow());
    std::vector<double> converted_image(src.fast() * src.slow());

    for (auto _ : state) {
        converted_image.assign(src.image_data().begin(), src.image_data().end());
        finder.standard_dispersion(converted_image, src.mask_data());
    }
}
BENCHMARK(BM_Standalone_dispersion_w_convert)->Unit(benchmark::kMillisecond);

static void BM_Standalone_dispersion(benchmark::State& state) {
    ImageSource<uint16_t> src;
    auto image = src.h5read_image();
    auto finder = StandaloneSpotfinder<double>(src.fast(), src.slow());
    std::vector<double> converted_image(src.fast() * src.slow());
    converted_image.assign(src.image_data().begin(), src.image_data().end());

    for (auto _ : state) {
        finder.standard_dispersion(converted_image, src.mask_data());
    }
}
BENCHMARK(BM_Standalone_dispersion)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
