/**
 * @file test_kabsch.cc
 * @brief Unit test for the GPU Kabsch transform foreground/background
 *        pixel classification
 *
 * Two families of tests share the same geometry/prediction inputs
 * (loadKabschInputs):
 *
 *  - ForegroundBackgroundPixelCounts{Dials,Ellipsoid} drive
 *    compute_kabsch_transform with a blank (all-zero) detector image, so the
 *    only thing tested is the per-pixel Kabsch foreground/background
 *    classification. Pixel sums are zero and unchecked; only the per-reflection
 *    foreground/background pixel COUNTS are compared to the baseline.
 *
 *  - IntensitySum{Dials,Ellipsoid} drive the same kernel over the REAL detector
 *    frames and mask, and check the per-reflection foreground intensity SUM. The
 *    kernel emits a raw foreground sum; the baseline subtracts a robust Tukey
 *    background. background.mean is the mean background per pixel, so the raw
 *    sum is recovered as intensity.sum.value + n_foreground * background.mean
 *    (no background model needed here).
 *
 * Both match the baseline by row index.
 *
 * Inputs (in FFS_INTEGRATE_TEST_DATA, fallback /scratch/ffs_integrate_test_data):
 *   - integrated_1_10.refl         -> s1, xyzcal.mm, bbox, miller_index
 *   - baseline_<algo>_1_10.refl    -> num_pixels.foreground / .background
 *                                      reference (row-aligned with the
 *                                      prediction table)
 *   - indexed_1_10.expt            -> beam, detector panel, goniometer, scan
 *
 * delta_b and delta_m are fixed test values (the indexed.expt used here
 * has no fitted profile model). If the data directory is missing the
 * test is skipped.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dx2/beam.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <tuple>
#include <vector>

#include "cuda_common.hpp"
#include "h5read.h"
#include "integrator.cuh"
#include "integrator/background.cuh"
#include "integrator/background.hpp"
#include "integrator/extent.hpp"
#include "kabsch.cuh"
#include "math/math_utils.cuh"
#include "math/vector3d.cuh"

namespace fs = std::filesystem;

/// HDF5 group path prefix for DIALS reflection table datasets
static const std::string G = "dials/processing/group_0/";

/**
 * @brief Geometry and per-reflection inputs shared by every Kabsch test.
 *
 * Loaded once from indexed_1_10.expt (geometry) and integrated_1_10.refl
 * (predictions/bboxes); the only thing that varies between tests is the
 * foreground algorithm and whether real image data is fed to the kernel.
 */
struct KabschInputs {
    // Geometry / scan
    DetectorParameters det_params;
    fastvec::Vector3D s0_vec;
    fastvec::Vector3D rot_axis_vec;
    std::vector<scalar_t> d_matrix_scalar;  // 3x3 flattened
    scalar_t wavelength;
    scalar_t osc_start;
    scalar_t osc_width;
    int image_range_start;
    int image_range_end;
    uint32_t width;
    uint32_t height;
    scalar_t delta_b;
    scalar_t delta_m;
    std::string image_template;  // path to the image file (nxs) for this scan

    // Per-reflection predictions
    size_t num_reflections;
    std::vector<fastvec::Vector3D> s1_vecs;
    std::vector<scalar_t> phi_vals;
    std::vector<BoundingBoxExtents> bboxes;
};

/// Recover per-reflection background pixel counts from the device histogram +
/// overflow buffers (sum of all bins plus the high-tail overflow). The kernel
/// no longer keeps a running background count; it is derived here.
static std::vector<uint32_t> background_counts(const DeviceBuffer<uint32_t> &d_hist,
                                               const DeviceBuffer<uint32_t> &d_overflow,
                                               size_t num_reflections) {
    std::vector<uint32_t> hist(num_reflections * NUM_BG_BINS);
    std::vector<uint32_t> overflow(num_reflections);
    d_hist.extract(hist.data());
    d_overflow.extract(overflow.data());
    std::vector<uint32_t> counts(num_reflections);
    for (size_t r = 0; r < num_reflections; ++r) {
        uint32_t total = overflow[r];
        for (int v = 0; v < NUM_BG_BINS; ++v) {
            total += hist[r * NUM_BG_BINS + v];
        }
        counts[r] = total;
    }
    return counts;
}

/// Parse indexed_1_10.expt + integrated_1_10.refl into kernel-facing inputs.
static KabschInputs loadKabschInputs(const fs::path &data_dir) {
    KabschInputs in;

    auto expt_file = data_dir / "indexed_1_10.expt";
    auto refl_file = data_dir / "integrated_1_10.refl";

    std::ifstream f(expt_file);
    auto elist_json = nlohmann::json::parse(f);
    Experiment expt(elist_json);
    const Panel &panel = expt.detector().panels()[0];
    const Scan &scan = expt.scan();
    MonochromaticBeam beam = std::get<MonochromaticBeam>(expt.beam());

    Eigen::Vector3d s0_eigen = beam.get_s0();
    Eigen::Vector3d rot_axis_eigen = expt.goniometer().get_rotation_axis();
    in.wavelength = static_cast<scalar_t>(beam.get_wavelength());
    auto oscillation = scan.get_oscillation();
    in.osc_start = static_cast<scalar_t>(oscillation[0]);
    in.osc_width = static_cast<scalar_t>(oscillation[1]);
    auto image_range = scan.get_image_range();
    in.image_range_start = image_range[0];
    in.image_range_end = image_range[1];

    Eigen::Matrix3d d_matrix_eigen = panel.get_d_matrix();
    in.d_matrix_scalar.resize(9);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            in.d_matrix_scalar[i * 3 + j] = static_cast<scalar_t>(d_matrix_eigen(i, j));

    in.det_params = make_detector_params(panel);

    in.s0_vec = fastvec::make_vector3d(static_cast<scalar_t>(s0_eigen[0]),
                                       static_cast<scalar_t>(s0_eigen[1]),
                                       static_cast<scalar_t>(s0_eigen[2]));
    in.rot_axis_vec = fastvec::make_vector3d(static_cast<scalar_t>(rot_axis_eigen[0]),
                                             static_cast<scalar_t>(rot_axis_eigen[1]),
                                             static_cast<scalar_t>(rot_axis_eigen[2]));

    auto image_size_mm = panel.get_image_size_mm();
    auto pixel_size = panel.get_pixel_size();
    in.width = static_cast<uint32_t>(std::round(image_size_mm[0] / pixel_size[0]));
    in.height = static_cast<uint32_t>(std::round(image_size_mm[1] / pixel_size[1]));

    // Fixed test values (indexed.expt has no fitted profile model)
    constexpr scalar_t N_SIGMA = scalar_t(3.0);
    in.delta_b = N_SIGMA * static_cast<scalar_t>(degrees_to_radians(0.03));
    in.delta_m = N_SIGMA * static_cast<scalar_t>(degrees_to_radians(0.1));

    // Image file for this scan (used by the real-image intensity tests)
    in.image_template = elist_json["imageset"][0]["template"].get<std::string>();

    // Predicted s1 vectors, calculated positions and bboxes
    auto refl_path = refl_file.string();
    auto s1_flat = read_array_from_h5_file<double>(refl_path, G + "s1");
    auto xyzcal_flat = read_array_from_h5_file<double>(refl_path, G + "xyzcal.mm");
    auto bbox_flat = read_array_from_h5_file<int32_t>(refl_path, G + "bbox");

    in.num_reflections = s1_flat.size() / 3;
    in.s1_vecs.resize(in.num_reflections);
    in.phi_vals.resize(in.num_reflections);
    in.bboxes.resize(in.num_reflections);
    for (size_t i = 0; i < in.num_reflections; ++i) {
        in.s1_vecs[i] =
          fastvec::make_vector3d(static_cast<scalar_t>(s1_flat[i * 3 + 0]),
                                 static_cast<scalar_t>(s1_flat[i * 3 + 1]),
                                 static_cast<scalar_t>(s1_flat[i * 3 + 2]));
        in.phi_vals[i] = static_cast<scalar_t>(xyzcal_flat[i * 3 + 2]);
        in.bboxes[i].x_min = bbox_flat[i * 6 + 0];
        in.bboxes[i].x_max = bbox_flat[i * 6 + 1];
        in.bboxes[i].y_min = bbox_flat[i * 6 + 2];
        in.bboxes[i].y_max = bbox_flat[i * 6 + 3];
        in.bboxes[i].z_min = bbox_flat[i * 6 + 4];
        in.bboxes[i].z_max = bbox_flat[i * 6 + 5];
    }

    return in;
}

class KabschTransformTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (const char *env = std::getenv("FFS_INTEGRATE_TEST_DATA")) {
            data_dir = env;
        } else {
            data_dir = "/scratch/ffs_integrate_test_data";
        }
        if (!fs::exists(data_dir)) {
            GTEST_SKIP() << "Integrator test data directory not found: " << data_dir
                         << " (set FFS_INTEGRATE_TEST_DATA to override)";
        }
    }

    fs::path data_dir;

    // Resolve the baseline reference (CPU integrator run with the SAME algorithm,
    // delta_b/delta_m and predictions). Overridable via FFS_KABSCH_BASELINE_REFERENCE.
    fs::path baselineReference(FGAlgorithm algo) const {
        if (const char *gp = std::getenv("FFS_KABSCH_BASELINE_REFERENCE")) {
            return gp;
        }
        return data_dir
               / ((algo == FGAlgorithm::Ellipsoid) ? "baseline_ellipsoid_1_10.refl"
                                                   : "baseline_dials_1_10.refl");
    }

    // Blank-image comparison of foreground/background pixel COUNTS.
    void RunPixelCountComparison(FGAlgorithm algo);
    // Real-image comparison of the foreground intensity SUM.
    void RunIntensitySumComparison(FGAlgorithm algo);
};

void KabschTransformTest::RunPixelCountComparison(FGAlgorithm algo) {
#pragma region Load inputs
    auto refl_file = data_dir / "integrated_1_10.refl";
    auto expt_file = data_dir / "indexed_1_10.expt";
    // Reference is the CPU baseline integrator run with the SAME algorithm,
    // the SAME delta_b/delta_m (sigma_b=0.03, sigma_m=0.1, n=3), and the
    // SAME predictions/bboxes (it consumes integrated_1_10.refl).
    // Regenerate with (one per algorithm):
    //   baseline_integrator integrated_1_10.refl indexed_1_10.expt \
    //     baseline_ellipsoid_1_10.refl --algorithm ellipsoid \
    //     --sigma_b 0.03 --sigma_m 0.1
    //   baseline_integrator integrated_1_10.refl indexed_1_10.expt \
    //     baseline_dials_1_10.refl --algorithm dials \
    //     --sigma_b 0.03 --sigma_m 0.1
    // Path is overridable via FFS_KABSCH_BASELINE_REFERENCE.
    fs::path reference_file = baselineReference(algo);
    ASSERT_TRUE(fs::exists(refl_file)) << "Missing input file: " << refl_file;
    ASSERT_TRUE(fs::exists(reference_file)) << "Missing input file: " << reference_file;
    ASSERT_TRUE(fs::exists(expt_file)) << "Missing input file: " << expt_file;

    KabschInputs in = loadKabschInputs(data_dir);
    size_t num_reflections = in.num_reflections;
    uint32_t width = in.width;
    uint32_t height = in.height;
    int image_range_start = in.image_range_start;
    int image_range_end = in.image_range_end;
    scalar_t delta_b = in.delta_b;
    scalar_t delta_m = in.delta_m;
    scalar_t wavelength = in.wavelength;
    scalar_t osc_start = in.osc_start;
    scalar_t osc_width = in.osc_width;
    DetectorParameters det_params = in.det_params;
    fastvec::Vector3D s0_vec = in.s0_vec;
    fastvec::Vector3D rot_axis_vec = in.rot_axis_vec;
    const std::vector<fastvec::Vector3D> &s1_vecs = in.s1_vecs;
    const std::vector<scalar_t> &phi_vals = in.phi_vals;
    const std::vector<BoundingBoxExtents> &bboxes = in.bboxes;
#pragma endregion Load inputs

#pragma region Allocate buffers
    // Allocate a blank (all-zero) detector image on the GPU, reused per image
    auto d_image = PitchedMalloc<pixel_t>(width, height);
    cudaMemset2D(
      d_image.get(), d_image.pitch_bytes(), 0, width * sizeof(pixel_t), height);
    cuda_throw_error();

    DeviceBuffer<scalar_t> d_d_matrix(9);
    DeviceBuffer<fastvec::Vector3D> d_s1_vectors(num_reflections);
    DeviceBuffer<scalar_t> d_phi_values(num_reflections);
    DeviceBuffer<BoundingBoxExtents> d_bboxes(num_reflections);
    d_d_matrix.assign(in.d_matrix_scalar.data());
    d_s1_vectors.assign(s1_vecs.data());
    d_phi_values.assign(phi_vals.data());
    d_bboxes.assign(bboxes.data());

    // Per-reflection accumulators (sums are unchecked; image is zero)
    DeviceBuffer<accumulator_t> d_fg_sum(num_reflections);
    DeviceBuffer<uint32_t> d_fg_count(num_reflections);
    // Background histogram (one bin per integer value) + overflow tail. The
    // background pixel count is recovered by summing these on the host.
    DeviceBuffer<uint32_t> d_bg_hist(num_reflections * NUM_BG_BINS);
    DeviceBuffer<uint32_t> d_bg_overflow(num_reflections);
    // Centre-of-mass accumulators: kernel writes unconditionally when any
    // foreground pixel is found, so valid device memory is required even
    // though this test only inspects the fg/bg pixel counts.
    DeviceBuffer<unsigned long long> d_intensity_times_x(num_reflections);
    DeviceBuffer<unsigned long long> d_intensity_times_y(num_reflections);
    DeviceBuffer<unsigned long long> d_intensity_times_z(num_reflections);
    cudaMemset(d_fg_sum.data(), 0, num_reflections * sizeof(accumulator_t));
    cudaMemset(d_fg_count.data(), 0, num_reflections * sizeof(uint32_t));
    cudaMemset(d_bg_hist.data(), 0, num_reflections * NUM_BG_BINS * sizeof(uint32_t));
    cudaMemset(d_bg_overflow.data(), 0, num_reflections * sizeof(uint32_t));
    cudaMemset(
      d_intensity_times_x.data(), 0, num_reflections * sizeof(unsigned long long));
    cudaMemset(
      d_intensity_times_y.data(), 0, num_reflections * sizeof(unsigned long long));
    cudaMemset(
      d_intensity_times_z.data(), 0, num_reflections * sizeof(unsigned long long));

    // All-valid detector mask (non-zero = valid) and a per-reflection success
    // flag. The kernel clears success on a masked or out-of-image foreground
    // pixel; with no mask and an image-sized grid, neither occurs here.
    DeviceBuffer<uint8_t> d_mask(static_cast<size_t>(width) * height);
    cudaMemset(d_mask.data(), 1, static_cast<size_t>(width) * height * sizeof(uint8_t));
    DeviceBuffer<uint8_t> d_success(num_reflections);
    cudaMemset(d_success.data(), 1, num_reflections * sizeof(uint8_t));
    cuda_throw_error();
#pragma endregion Allocate buffers

#pragma region Dispatch
    // For each image in the scan, gather reflections whose bbox covers it
    // (z_min <= image_num < z_max) and dispatch one kernel call. Counts
    // accumulate across images via atomic adds inside the kernel.
    DeviceBuffer<size_t> d_reflection_indices(num_reflections);
    std::vector<size_t> refl_indices_this_image;
    refl_indices_this_image.reserve(num_reflections);

    for (int image_num = image_range_start; image_num <= image_range_end; ++image_num) {
        refl_indices_this_image.clear();
        // Largest (w+1)*(h+1) corner grid on this image, sizing the kernel's
        // dynamic shared-memory corner tile (as the integrator does).
        uint32_t max_corner_tile = 0;
        for (size_t i = 0; i < num_reflections; ++i) {
            if (bboxes[i].z_min <= image_num && image_num < bboxes[i].z_max) {
                refl_indices_this_image.push_back(i);
                const uint32_t corner_grid =
                  static_cast<uint32_t>(bboxes[i].x_max - bboxes[i].x_min + 1)
                  * static_cast<uint32_t>(bboxes[i].y_max - bboxes[i].y_min + 1);
                max_corner_tile = std::max(max_corner_tile, corner_grid);
            }
        }
        if (refl_indices_this_image.empty()) continue;

        cudaMemcpy(d_reflection_indices.data(),
                   refl_indices_this_image.data(),
                   refl_indices_this_image.size() * sizeof(size_t),
                   cudaMemcpyHostToDevice);
        cuda_throw_error();

        compute_kabsch_transform(d_image.get(),
                                 d_image.pitch_bytes(),
                                 width,
                                 height,
                                 image_num,
                                 d_d_matrix.data(),
                                 wavelength,
                                 det_params,
                                 osc_start,
                                 osc_width,
                                 image_range_start,
                                 s0_vec,
                                 rot_axis_vec,
                                 d_s1_vectors.data(),
                                 d_phi_values.data(),
                                 d_bboxes.data(),
                                 d_reflection_indices.data(),
                                 refl_indices_this_image.size(),
                                 d_mask.data(),
                                 max_corner_tile,
                                 delta_b,
                                 delta_m,
                                 algo,
                                 d_fg_sum.data(),
                                 d_fg_count.data(),
                                 d_bg_hist.data(),
                                 d_bg_overflow.data(),
                                 d_intensity_times_x.data(),
                                 d_intensity_times_y.data(),
                                 d_intensity_times_z.data(),
                                 d_success.data(),
                                 nullptr);
    }
    cudaDeviceSynchronize();
    cuda_throw_error();
#pragma endregion Dispatch

#pragma region Compare counts
    std::vector<uint32_t> h_fg_count(num_reflections);
    std::vector<uint32_t> h_bg_count =
      background_counts(d_bg_hist, d_bg_overflow, num_reflections);
    d_fg_count.extract(h_fg_count.data());

    // Load baseline foreground/background pixel counts. The baseline writes
    // rows in the SAME order as its input (integrated_1_10.refl), which is
    // exactly what refl_file feeds the kernel, so we match by ROW INDEX.
    auto reference_path = reference_file.string();
    auto expected_fg_int =
      read_array_from_h5_file<int32_t>(reference_path, G + "num_pixels.foreground");
    auto expected_bg_int =
      read_array_from_h5_file<int32_t>(reference_path, G + "num_pixels.background");
    size_t num_expected = expected_fg_int.size();
    ASSERT_EQ(expected_bg_int.size(), num_expected);
    ASSERT_EQ(num_expected, num_reflections)
      << "Baseline reference row count must match the prediction table; "
         "regenerate baseline_dials_1_10.refl from integrated_1_10.refl";

    // A reflection is comparable only when the kernel and the baseline saw
    // the SAME pixel population. The baseline reads real images and applies
    // the detector mask, so it drops masked / off-edge pixels; the kernel
    // runs an unmasked blank image and counts the whole in-image bbox. When
    // the totals (fg+bg) agree, no pixel was masked/clipped for that
    // reflection and the foreground/background split must match exactly.
    // When they differ, the difference is entirely the detector mask, which
    // this test does not model, so the reflection is excluded.
    size_t exact_matches = 0;
    size_t mismatches = 0;
    size_t mask_excluded = 0;

    const char *comp_names[] = {"n_fg", "n_bg"};
    std::array<double, 2> sum_abs_err = {};
    std::array<double, 2> sum_signed_err = {};
    std::array<int64_t, 2> max_abs_err = {};
    std::array<std::map<int, int>, 2> diff_hist;

    for (size_t i = 0; i < num_reflections; ++i) {
        std::array<int64_t, 2> got = {static_cast<int64_t>(h_fg_count[i]),
                                      static_cast<int64_t>(h_bg_count[i])};
        std::array<int64_t, 2> exp = {static_cast<int64_t>(expected_fg_int[i]),
                                      static_cast<int64_t>(expected_bg_int[i])};

        if (got[0] + got[1] != exp[0] + exp[1]) {
            mask_excluded++;
            continue;
        }

        if (got[0] == exp[0] && got[1] == exp[1]) {
            exact_matches++;
        } else {
            mismatches++;
            for (int c = 0; c < 2; ++c) {
                int64_t diff = got[c] - exp[c];
                int64_t absdiff = std::abs(diff);
                sum_abs_err[c] += static_cast<double>(absdiff);
                sum_signed_err[c] += static_cast<double>(diff);
                max_abs_err[c] = std::max(max_abs_err[c], absdiff);
                diff_hist[c][static_cast<int>(diff)]++;
            }
        }
    }

    size_t compared = exact_matches + mismatches;

    const char *algo_name = (algo == FGAlgorithm::Ellipsoid) ? "Ellipsoid" : "Dials";
    std::cout << "\n=== Kabsch Pixel Count Comparison Summary (" << algo_name
              << ") ===\n";
    std::cout << "  Reflections compared : " << compared << " (of " << num_reflections
              << ")\n";
    std::cout << "  Mask-excluded (fg+bg differs, not modelled) : " << mask_excluded
              << "\n";
    std::cout << "  Exact matches        : " << exact_matches << " ("
              << (compared ? 100.0 * exact_matches / compared : 0) << "%)\n";
    std::cout << "  Mismatches           : " << mismatches << "\n";

    if (mismatches > 0) {
        std::cout << "\n  Per-component error breakdown (over " << mismatches
                  << " mismatched reflections):\n";
        std::cout << "  Component  MeanAbsErr  MeanSignedErr  MaxAbsErr\n";
        for (int c = 0; c < 2; ++c) {
            double mean_abs = sum_abs_err[c] / mismatches;
            double mean_signed = sum_signed_err[c] / mismatches;
            std::cout << "  " << std::setw(9) << std::left << comp_names[c] << "  "
                      << std::setw(10) << std::fixed << std::setprecision(3) << mean_abs
                      << "  " << std::setw(13) << std::fixed << std::setprecision(3)
                      << mean_signed << "  " << std::setw(9) << max_abs_err[c] << "\n";
        }

        constexpr int TOP_N = 5;
        std::cout << "\n  Top-" << TOP_N
                  << " signed differences per component (computed - expected):\n";
        for (int c = 0; c < 2; ++c) {
            if (diff_hist[c].empty()) continue;
            std::vector<std::pair<int, int>> entries(diff_hist[c].begin(),
                                                     diff_hist[c].end());
            std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
                return b.second < a.second;
            });
            std::cout << "  " << comp_names[c] << ":\n";
            int shown = 0;
            for (const auto &[diff, count] : entries) {
                if (shown++ >= TOP_N) break;
                std::cout << "    " << std::setw(6) << std::right << diff << " : "
                          << count << "\n";
            }
        }
    }
    std::cout << "=============================================\n";

    // Every reflection with an identical pixel population must produce an
    // identical foreground/background split as the baseline CPU algorithm.
    EXPECT_GT(compared, 0u) << "No comparable reflections (all mask-excluded?)";
    EXPECT_EQ(mismatches, 0) << mismatches << " of " << compared
                             << " comparable reflections diverged from the "
                                "baseline "
                             << algo_name << " reference";
#pragma endregion Compare counts
}

TEST_F(KabschTransformTest, ForegroundBackgroundPixelCountsDials) {
    RunPixelCountComparison(FGAlgorithm::Dials);
}

TEST_F(KabschTransformTest, ForegroundBackgroundPixelCountsEllipsoid) {
    RunPixelCountComparison(FGAlgorithm::Ellipsoid);
}

/**
 * @brief Compare the kernel's per-reflection foreground intensity sum against
 *        the baseline, driven over the real detector frames.
 *
 * Unlike the count tests above (which use a blank image), this feeds the actual
 * scan frames and the real detector mask, so the foreground summation itself is
 * exercised. The kernel accumulates a raw foreground sum, whereas the baseline
 * subtracts a robust Tukey background, so its intensity.sum.value is not a raw
 * sum. The raw sum is recovered exactly as intensity.sum.value + background.mean
 * (the baseline stores I = fg_sum - background.mean), so no background model is
 * needed here.
 */
void KabschTransformTest::RunIntensitySumComparison(FGAlgorithm algo) {
#pragma region Load inputs
    auto refl_file = data_dir / "integrated_1_10.refl";
    auto expt_file = data_dir / "indexed_1_10.expt";
    fs::path reference_file = baselineReference(algo);
    ASSERT_TRUE(fs::exists(refl_file)) << "Missing input file: " << refl_file;
    ASSERT_TRUE(fs::exists(reference_file)) << "Missing input file: " << reference_file;
    ASSERT_TRUE(fs::exists(expt_file)) << "Missing input file: " << expt_file;

    KabschInputs in = loadKabschInputs(data_dir);
    size_t num_reflections = in.num_reflections;
    uint32_t width = in.width;
    uint32_t height = in.height;

    // Open the real image file referenced by the experiment.
    ASSERT_TRUE(fs::exists(in.image_template))
      << "Image file referenced by expt not found: " << in.image_template;
    H5Read reader(in.image_template);
    int num_images = static_cast<int>(reader.get_number_of_images());
    auto shape = reader.image_shape();  // {slow, fast}
    ASSERT_EQ(shape[1], width) << "Image fast dimension disagrees with detector";
    ASSERT_EQ(shape[0], height) << "Image slow dimension disagrees with detector";
#pragma endregion Load inputs

#pragma region Allocate buffers
    // Static per-reflection inputs.
    auto d_image = PitchedMalloc<pixel_t>(width, height);
    DeviceBuffer<scalar_t> d_d_matrix(9);
    DeviceBuffer<fastvec::Vector3D> d_s1_vectors(num_reflections);
    DeviceBuffer<scalar_t> d_phi_values(num_reflections);
    DeviceBuffer<BoundingBoxExtents> d_bboxes(num_reflections);
    d_d_matrix.assign(in.d_matrix_scalar.data());
    d_s1_vectors.assign(in.s1_vecs.data());
    d_phi_values.assign(in.phi_vals.data());
    d_bboxes.assign(in.bboxes.data());

    // Per-reflection accumulators (zeroed; the kernel atomically adds into them).
    DeviceBuffer<accumulator_t> d_fg_sum(num_reflections);
    DeviceBuffer<uint32_t> d_fg_count(num_reflections);
    DeviceBuffer<uint32_t> d_bg_hist(num_reflections * NUM_BG_BINS);
    DeviceBuffer<uint32_t> d_bg_overflow(num_reflections);
    DeviceBuffer<unsigned long long> d_itx(num_reflections);
    DeviceBuffer<unsigned long long> d_ity(num_reflections);
    DeviceBuffer<unsigned long long> d_itz(num_reflections);
    cudaMemset(d_fg_sum.data(), 0, num_reflections * sizeof(accumulator_t));
    cudaMemset(d_fg_count.data(), 0, num_reflections * sizeof(uint32_t));
    cudaMemset(d_bg_hist.data(), 0, num_reflections * NUM_BG_BINS * sizeof(uint32_t));
    cudaMemset(d_bg_overflow.data(), 0, num_reflections * sizeof(uint32_t));
    cudaMemset(d_itx.data(), 0, num_reflections * sizeof(unsigned long long));
    cudaMemset(d_ity.data(), 0, num_reflections * sizeof(unsigned long long));
    cudaMemset(d_itz.data(), 0, num_reflections * sizeof(unsigned long long));

    // Real detector mask (non-zero = valid); all-valid fallback if absent.
    std::vector<uint8_t> h_mask(static_cast<size_t>(width) * height, 1);
    if (auto mask_opt = reader.get_mask()) {
        std::copy(mask_opt->begin(), mask_opt->end(), h_mask.begin());
    }
    DeviceBuffer<uint8_t> d_mask(static_cast<size_t>(width) * height);
    d_mask.assign(h_mask.data());
    DeviceBuffer<uint8_t> d_success(num_reflections);
    cudaMemset(d_success.data(), 1, num_reflections * sizeof(uint8_t));
    cuda_throw_error();
#pragma endregion Allocate buffers

#pragma region Dispatch
    // Iterate frames in 0-based frame space (== bbox z space, == baseline). Load
    // the real pixels for each frame and dispatch one kernel call.
    std::vector<pixel_t> host_image(static_cast<size_t>(width) * height);
    DeviceBuffer<size_t> d_reflection_indices(num_reflections);
    std::vector<size_t> refl_indices_this_image;
    refl_indices_this_image.reserve(num_reflections);

    for (int image_num = 0; image_num < num_images; ++image_num) {
        refl_indices_this_image.clear();
        // Largest (w+1)*(h+1) corner grid on this image, sizing the kernel's
        // dynamic shared-memory corner tile (as the integrator does).
        uint32_t max_corner_tile = 0;
        for (size_t i = 0; i < num_reflections; ++i) {
            if (in.bboxes[i].z_min <= image_num && image_num < in.bboxes[i].z_max) {
                refl_indices_this_image.push_back(i);
                const uint32_t corner_grid =
                  static_cast<uint32_t>(in.bboxes[i].x_max - in.bboxes[i].x_min + 1)
                  * static_cast<uint32_t>(in.bboxes[i].y_max - in.bboxes[i].y_min + 1);
                max_corner_tile = std::max(max_corner_tile, corner_grid);
            }
        }
        if (refl_indices_this_image.empty()) continue;

        // is_image_available() lazily opens the frame's dataset handle (via the
        // raw-chunk path); get_image_into() relies on it already being open, so
        // this call is required before reading, as the baseline/integrator do.
        ASSERT_TRUE(reader.is_image_available(static_cast<size_t>(image_num)))
          << "Image frame " << image_num << " not available in " << in.image_template;
        reader.get_image_into(static_cast<size_t>(image_num), host_image.data());
        cudaMemcpy2D(d_image.get(),
                     d_image.pitch_bytes(),
                     host_image.data(),
                     width * sizeof(pixel_t),
                     width * sizeof(pixel_t),
                     height,
                     cudaMemcpyHostToDevice);
        cuda_throw_error();

        cudaMemcpy(d_reflection_indices.data(),
                   refl_indices_this_image.data(),
                   refl_indices_this_image.size() * sizeof(size_t),
                   cudaMemcpyHostToDevice);
        cuda_throw_error();

        compute_kabsch_transform(d_image.get(),
                                 d_image.pitch_bytes(),
                                 width,
                                 height,
                                 image_num,
                                 d_d_matrix.data(),
                                 in.wavelength,
                                 in.det_params,
                                 in.osc_start,
                                 in.osc_width,
                                 in.image_range_start,
                                 in.s0_vec,
                                 in.rot_axis_vec,
                                 d_s1_vectors.data(),
                                 d_phi_values.data(),
                                 d_bboxes.data(),
                                 d_reflection_indices.data(),
                                 refl_indices_this_image.size(),
                                 d_mask.data(),
                                 max_corner_tile,
                                 in.delta_b,
                                 in.delta_m,
                                 algo,
                                 d_fg_sum.data(),
                                 d_fg_count.data(),
                                 d_bg_hist.data(),
                                 d_bg_overflow.data(),
                                 d_itx.data(),
                                 d_ity.data(),
                                 d_itz.data(),
                                 d_success.data(),
                                 nullptr);
    }
    cudaDeviceSynchronize();
    cuda_throw_error();
#pragma endregion Dispatch

#pragma region Reduce background
    std::vector<accumulator_t> h_fg_sum(num_reflections);
    std::vector<uint32_t> h_fg_count(num_reflections);
    std::vector<uint32_t> h_bg_count =
      background_counts(d_bg_hist, d_bg_overflow, num_reflections);
    d_fg_sum.extract(h_fg_sum.data());
    d_fg_count.extract(h_fg_count.data());

    // Reduce the per-reflection background histograms on the device (the same
    // Tukey/IQR path the integrator uses) to compare the GPU background estimate
    // against the baseline.
    DeviceBuffer<double> d_bg_mean(num_reflections);
    DeviceBuffer<double> d_bg_sum_value(num_reflections);
    DeviceBuffer<uint32_t> d_bg_count_r(num_reflections);
    DeviceBuffer<uint8_t> d_bg_success(num_reflections);
    compute_background(BackgroundModel::Constant,
                       d_bg_hist.data(),
                       d_bg_overflow.data(),
                       num_reflections,
                       d_bg_mean.data(),
                       d_bg_sum_value.data(),
                       d_bg_count_r.data(),
                       d_bg_success.data(),
                       nullptr);
    cudaDeviceSynchronize();
    cuda_throw_error();
    std::vector<double> h_bg_mean(num_reflections);
    std::vector<double> h_bg_sum_value(num_reflections);
    std::vector<uint8_t> h_bg_success(num_reflections);
    d_bg_mean.extract(h_bg_mean.data());
    d_bg_sum_value.extract(h_bg_sum_value.data());
    d_bg_success.extract(h_bg_success.data());
#pragma endregion Reduce background

#pragma region Load baseline
    // Baseline columns (row-aligned with the prediction table).
    auto reference_path = reference_file.string();
    auto expected_fg_int =
      read_array_from_h5_file<int32_t>(reference_path, G + "num_pixels.foreground");
    auto expected_bg_int =
      read_array_from_h5_file<int32_t>(reference_path, G + "num_pixels.background");
    auto expected_intensity =
      read_array_from_h5_file<double>(reference_path, G + "intensity.sum.value");
    auto expected_bg_mean =
      read_array_from_h5_file<double>(reference_path, G + "background.mean");
    auto expected_bg_sum =
      read_array_from_h5_file<double>(reference_path, G + "background.sum.value");
    ASSERT_EQ(expected_fg_int.size(), num_reflections);
    ASSERT_EQ(expected_bg_int.size(), num_reflections);
    ASSERT_EQ(expected_intensity.size(), num_reflections);
    ASSERT_EQ(expected_bg_mean.size(), num_reflections);
    ASSERT_EQ(expected_bg_sum.size(), num_reflections);
#pragma endregion Load baseline

#pragma region Compare results
    // Foreground sum parity requires the kernel and baseline to have selected the
    // identical foreground pixel set, which holds when both the fg and bg counts
    // match. For those, the raw foreground sums must be equal:
    //   kernel fg_sum == intensity.sum.value + background.mean.
    // Background parity depends only on the background population, so it is gated
    // on the bg count alone: a reflection whose foreground was clipped by the
    // mask can still have an identical background worth verifying.
    size_t compared = 0;
    size_t mismatches = 0;
    size_t excluded = 0;
    double sum_abs_err = 0.0;
    double max_abs_err = 0.0;
    // Background estimate parity (device Tukey reduction vs baseline).
    size_t bg_compared = 0;
    size_t bg_mismatches = 0;
    double bg_max_abs_err = 0.0;

    for (size_t i = 0; i < num_reflections; ++i) {
        // Background parity: same background pixel population (bg counts match)
        // => the device Tukey reduction must reproduce the baseline
        // background.mean and background.sum.value.
        if (static_cast<int64_t>(h_bg_count[i]) == expected_bg_int[i]) {
            bg_compared++;
            double bg_mean_err = std::abs(h_bg_mean[i] - expected_bg_mean[i]);
            double bg_sum_err = std::abs(h_bg_sum_value[i] - expected_bg_sum[i]);
            double bg_mean_tol = 1e-5 * std::abs(expected_bg_mean[i]) + 1e-4;
            double bg_sum_tol = 1e-5 * std::abs(expected_bg_sum[i]) + 1e-3;
            bg_max_abs_err = std::max(bg_max_abs_err, bg_mean_err);
            if (!h_bg_success[i] || bg_mean_err > bg_mean_tol
                || bg_sum_err > bg_sum_tol) {
                bg_mismatches++;
                if (bg_mismatches <= 10) {
                    std::cout << "  background mismatch refl " << i
                              << ": gpu_mean=" << h_bg_mean[i]
                              << " baseline_mean=" << expected_bg_mean[i]
                              << " gpu_sum=" << h_bg_sum_value[i]
                              << " baseline_sum=" << expected_bg_sum[i]
                              << " success=" << static_cast<int>(h_bg_success[i])
                              << "\n";
                }
            }
        }

        // Foreground sum parity needs the identical foreground pixel set, which
        // requires both the fg and bg counts to match.
        if (static_cast<int64_t>(h_fg_count[i]) != expected_fg_int[i]
            || static_cast<int64_t>(h_bg_count[i]) != expected_bg_int[i]) {
            excluded++;
            continue;
        }
        compared++;

        // Undo the baseline's background subtraction to get its raw foreground
        // sum. background.mean is the mean background per pixel, and the
        // baseline stored intensity.sum.value = fg_sum - n_fg * background.mean,
        // so adding n_fg * background.mean back cancels the (robust Tukey)
        // background term and leaves the same raw Σ over foreground pixels the
        // kernel produces. (n_fg == expected_fg_int[i] for the compared reflections.)
        double baseline_foreground_sum =
          expected_intensity[i]
          + expected_bg_mean[i] * static_cast<double>(expected_fg_int[i]);
        double kernel_foreground_sum = static_cast<double>(h_fg_sum[i]);
        double abs_err = std::abs(kernel_foreground_sum - baseline_foreground_sum);
        double tol = 1e-6 * std::abs(baseline_foreground_sum) + 1e-3;
        sum_abs_err += abs_err;
        max_abs_err = std::max(max_abs_err, abs_err);
        if (abs_err > tol) {
            mismatches++;
            if (mismatches <= 10) {
                std::cout << "  fg_sum mismatch refl " << i
                          << ": kernel=" << kernel_foreground_sum
                          << " baseline=" << baseline_foreground_sum
                          << " (I=" << expected_intensity[i]
                          << " + n_fg*bg_mean=" << expected_fg_int[i] << "*"
                          << expected_bg_mean[i] << "), |err|=" << abs_err << "\n";
            }
        }
    }

#pragma endregion Compare results

#pragma region Assertions
    const char *algo_name = (algo == FGAlgorithm::Ellipsoid) ? "Ellipsoid" : "Dials";
    std::cout << "\n=== Kabsch Foreground Intensity Sum Comparison (" << algo_name
              << ") ===\n";
    std::cout << "  Reflections compared : " << compared << " (of " << num_reflections
              << ")\n";
    std::cout << "  Excluded (counts differ) : " << excluded << "\n";
    std::cout << "  Mismatches           : " << mismatches << "\n";
    if (compared > 0) {
        std::cout << "  Mean |err|           : " << (sum_abs_err / compared) << "\n";
        std::cout << "  Max  |err|           : " << max_abs_err << "\n";
    }
    std::cout << "  Background compared  : " << bg_compared << " (of "
              << num_reflections << ")\n";
    std::cout << "  Background mismatches : " << bg_mismatches << " (max |mean err| "
              << bg_max_abs_err << ")\n";
    std::cout << "=============================================\n";

    EXPECT_GT(compared, 0u) << "No comparable reflections (counts never matched)";
    EXPECT_EQ(mismatches, 0u)
      << mismatches << " of " << compared
      << " comparable reflections had a foreground sum differing from the baseline "
      << algo_name << " reference";
    EXPECT_GT(bg_compared, 0u)
      << "No reflections with matching background counts to compare";
    EXPECT_EQ(bg_mismatches, 0u)
      << bg_mismatches << " of " << bg_compared
      << " reflections with matching background counts had a device Tukey background "
         "differing from the baseline "
      << algo_name << " reference";
#pragma endregion Assertions
}

TEST_F(KabschTransformTest, IntensitySumDials) {
    RunIntensitySumComparison(FGAlgorithm::Dials);
}

TEST_F(KabschTransformTest, IntensitySumEllipsoid) {
    RunIntensitySumComparison(FGAlgorithm::Ellipsoid);
}
