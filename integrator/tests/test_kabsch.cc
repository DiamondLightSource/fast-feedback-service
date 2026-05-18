/**
 * @file test_kabsch.cc
 * @brief Unit test for the GPU Kabsch transform foreground/background
 *        pixel classification
 *
 * Drives compute_kabsch_transform across every image in the scan with a
 * blank (all-zero) detector image, so the only thing being exercised is
 * the per-pixel Kabsch-ellipsoid foreground/background classification.
 * Pixel sums are necessarily zero and are not checked; only the
 * accumulated per-reflection foreground and background pixel COUNTS are
 * compared against a CPU baseline reference (see reference resolution
 * below), matched by row index.
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
#include "integrator.cuh"
#include "integrator/extent.hpp"
#include "kabsch.cuh"
#include "math/math_utils.cuh"
#include "math/vector3d.cuh"

namespace fs = std::filesystem;

/// HDF5 group path prefix for DIALS reflection table datasets
static const std::string G = "dials/processing/group_0/";

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

    // Shared comparison driver; the only thing that varies between the
    // DIALS and Ellipsoid tests is the foreground-classification algorithm.
    void RunPixelCountComparison(FGAlgorithm algo);
};

void KabschTransformTest::RunPixelCountComparison(FGAlgorithm algo) {
    auto refl_file = data_dir / "integrated_1_10.refl";
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
    // Path is overridable via FFS_KABSCH_BASELINE_REFERENCE; default falls back
    // to the build tree (data_dir is typically read-only test data).
    const std::string reference_name = (algo == FGAlgorithm::Ellipsoid)
                                         ? "baseline_ellipsoid_1_10.refl"
                                         : "baseline_dials_1_10.refl";
    fs::path reference_file;
    if (const char *gp = std::getenv("FFS_KABSCH_BASELINE_REFERENCE")) {
        reference_file = gp;
    } else if (fs::exists(data_dir / reference_name)) {
        reference_file = data_dir / reference_name;
    } else {
        reference_file = fs::path("../../test_references") / reference_name;
    }
    auto expt_file = data_dir / "indexed_1_10.expt";
    ASSERT_TRUE(fs::exists(refl_file)) << "Missing input file: " << refl_file;
    ASSERT_TRUE(fs::exists(reference_file)) << "Missing input file: " << reference_file;
    ASSERT_TRUE(fs::exists(expt_file)) << "Missing input file: " << expt_file;

    // Extract beam, detector panel, goniometer, and scan from the experiment
    std::ifstream f(expt_file);
    auto elist_json = nlohmann::json::parse(f);
    Experiment<MonochromaticBeam> expt(elist_json);
    const Panel &panel = expt.detector().panels()[0];
    const Scan &scan = expt.scan();
    MonochromaticBeam beam = expt.beam();

    Eigen::Vector3d s0_eigen = beam.get_s0();
    Eigen::Vector3d rot_axis_eigen = expt.goniometer().get_rotation_axis();
    scalar_t wavelength = static_cast<scalar_t>(beam.get_wavelength());
    auto oscillation = scan.get_oscillation();
    scalar_t osc_start = static_cast<scalar_t>(oscillation[0]);
    scalar_t osc_width = static_cast<scalar_t>(oscillation[1]);
    auto image_range = scan.get_image_range();
    int image_range_start = image_range[0];
    int image_range_end = image_range[1];

    Eigen::Matrix3d d_matrix_eigen = panel.get_d_matrix();
    std::vector<scalar_t> d_matrix_scalar(9);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            d_matrix_scalar[i * 3 + j] = static_cast<scalar_t>(d_matrix_eigen(i, j));

    DetectorParameters det_params = make_detector_params(panel);

    fastvec::Vector3D s0_vec =
      fastvec::make_vector3d(static_cast<scalar_t>(s0_eigen[0]),
                             static_cast<scalar_t>(s0_eigen[1]),
                             static_cast<scalar_t>(s0_eigen[2]));
    fastvec::Vector3D rot_axis_vec =
      fastvec::make_vector3d(static_cast<scalar_t>(rot_axis_eigen[0]),
                             static_cast<scalar_t>(rot_axis_eigen[1]),
                             static_cast<scalar_t>(rot_axis_eigen[2]));

    auto image_size_mm = panel.get_image_size_mm();
    auto pixel_size = panel.get_pixel_size();
    uint32_t width =
      static_cast<uint32_t>(std::round(image_size_mm[0] / pixel_size[0]));
    uint32_t height =
      static_cast<uint32_t>(std::round(image_size_mm[1] / pixel_size[1]));

    // Fixed test values
    constexpr scalar_t N_SIGMA = scalar_t(3.0);
    scalar_t delta_b = N_SIGMA * static_cast<scalar_t>(degrees_to_radians(0.03));
    scalar_t delta_m = N_SIGMA * static_cast<scalar_t>(degrees_to_radians(0.1));

    // Load predicted s₁ vectors, calculated positions, bboxes and Miller indices
    auto refl_path = refl_file.string();
    auto s1_flat = read_array_from_h5_file<double>(refl_path, G + "s1");
    auto xyzcal_flat = read_array_from_h5_file<double>(refl_path, G + "xyzcal.mm");
    auto bbox_flat = read_array_from_h5_file<int32_t>(refl_path, G + "bbox");
    auto hkl_flat = read_array_from_h5_file<int32_t>(refl_path, G + "miller_index");

    size_t num_reflections = s1_flat.size() / 3;
    ASSERT_EQ(xyzcal_flat.size(), num_reflections * 3);
    ASSERT_EQ(bbox_flat.size(), num_reflections * 6);
    ASSERT_EQ(hkl_flat.size(), num_reflections * 3);

    // Convert host data into kernel-facing types
    std::vector<fastvec::Vector3D> s1_vecs(num_reflections);
    std::vector<scalar_t> phi_vals(num_reflections);
    std::vector<BoundingBoxExtents> bboxes(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        s1_vecs[i] = fastvec::make_vector3d(static_cast<scalar_t>(s1_flat[i * 3 + 0]),
                                            static_cast<scalar_t>(s1_flat[i * 3 + 1]),
                                            static_cast<scalar_t>(s1_flat[i * 3 + 2]));
        phi_vals[i] = static_cast<scalar_t>(xyzcal_flat[i * 3 + 2]);
        bboxes[i].x_min = bbox_flat[i * 6 + 0];
        bboxes[i].x_max = bbox_flat[i * 6 + 1];
        bboxes[i].y_min = bbox_flat[i * 6 + 2];
        bboxes[i].y_max = bbox_flat[i * 6 + 3];
        bboxes[i].z_min = bbox_flat[i * 6 + 4];
        bboxes[i].z_max = bbox_flat[i * 6 + 5];
    }

    // Allocate a blank (all-zero) detector image on the GPU, reused per image
    auto d_image = PitchedMalloc<pixel_t>(width, height);
    cudaMemset2D(
      d_image.get(), d_image.pitch_bytes(), 0, width * sizeof(pixel_t), height);
    cuda_throw_error();

    DeviceBuffer<scalar_t> d_d_matrix(9);
    DeviceBuffer<fastvec::Vector3D> d_s1_vectors(num_reflections);
    DeviceBuffer<scalar_t> d_phi_values(num_reflections);
    DeviceBuffer<BoundingBoxExtents> d_bboxes(num_reflections);
    d_d_matrix.assign(d_matrix_scalar.data());
    d_s1_vectors.assign(s1_vecs.data());
    d_phi_values.assign(phi_vals.data());
    d_bboxes.assign(bboxes.data());

    // Per-reflection accumulators (sums are unchecked; image is zero)
    DeviceBuffer<accumulator_t> d_fg_sum(num_reflections);
    DeviceBuffer<uint32_t> d_fg_count(num_reflections);
    DeviceBuffer<accumulator_t> d_bg_sum(num_reflections);
    DeviceBuffer<uint32_t> d_bg_count(num_reflections);
    // Centre-of-mass accumulators: kernel writes unconditionally when any
    // foreground pixel is found, so valid device memory is required even
    // though this test only inspects the fg/bg pixel counts.
    DeviceBuffer<unsigned long long> d_intensity_times_x(num_reflections);
    DeviceBuffer<unsigned long long> d_intensity_times_y(num_reflections);
    DeviceBuffer<unsigned long long> d_intensity_times_z(num_reflections);
    cudaMemset(d_fg_sum.data(), 0, num_reflections * sizeof(accumulator_t));
    cudaMemset(d_fg_count.data(), 0, num_reflections * sizeof(uint32_t));
    cudaMemset(d_bg_sum.data(), 0, num_reflections * sizeof(accumulator_t));
    cudaMemset(d_bg_count.data(), 0, num_reflections * sizeof(uint32_t));
    cudaMemset(
      d_intensity_times_x.data(), 0, num_reflections * sizeof(unsigned long long));
    cudaMemset(
      d_intensity_times_y.data(), 0, num_reflections * sizeof(unsigned long long));
    cudaMemset(
      d_intensity_times_z.data(), 0, num_reflections * sizeof(unsigned long long));
    cuda_throw_error();

    // For each image in the scan, gather reflections whose bbox covers it
    // (z_min <= image_num < z_max) and dispatch one kernel call. Counts
    // accumulate across images via atomic adds inside the kernel.
    DeviceBuffer<size_t> d_reflection_indices(num_reflections);
    std::vector<size_t> refl_indices_this_image;
    refl_indices_this_image.reserve(num_reflections);

    for (int image_num = image_range_start; image_num <= image_range_end; ++image_num) {
        refl_indices_this_image.clear();
        for (size_t i = 0; i < num_reflections; ++i) {
            if (bboxes[i].z_min <= image_num && image_num < bboxes[i].z_max) {
                refl_indices_this_image.push_back(i);
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
                                 delta_b,
                                 delta_m,
                                 algo,
                                 d_fg_sum.data(),
                                 d_fg_count.data(),
                                 d_bg_sum.data(),
                                 d_bg_count.data(),
                                 d_intensity_times_x.data(),
                                 d_intensity_times_y.data(),
                                 d_intensity_times_z.data(),
                                 nullptr,
                                 nullptr);
    }
    cudaDeviceSynchronize();
    cuda_throw_error();

    std::vector<uint32_t> h_fg_count(num_reflections);
    std::vector<uint32_t> h_bg_count(num_reflections);
    d_fg_count.extract(h_fg_count.data());
    d_bg_count.extract(h_bg_count.data());

    // Load baseline foreground/background pixel counts. The baseline writes
    // rows in the SAME order as its input (integrated_1_10.refl), which is
    // exactly what refl_file feeds the kernel, so we match by ROW INDEX.
    auto reference_path = reference_file.string();
    auto exp_fg_int =
      read_array_from_h5_file<int32_t>(reference_path, G + "num_pixels.foreground");
    auto exp_bg_int =
      read_array_from_h5_file<int32_t>(reference_path, G + "num_pixels.background");
    size_t num_expected = exp_fg_int.size();
    ASSERT_EQ(exp_bg_int.size(), num_expected);
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
        std::array<int64_t, 2> exp = {static_cast<int64_t>(exp_fg_int[i]),
                                      static_cast<int64_t>(exp_bg_int[i])};

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
}

TEST_F(KabschTransformTest, ForegroundBackgroundPixelCountsDials) {
    RunPixelCountComparison(FGAlgorithm::Dials);
}

TEST_F(KabschTransformTest, ForegroundBackgroundPixelCountsEllipsoid) {
    RunPixelCountComparison(FGAlgorithm::Ellipsoid);
}
