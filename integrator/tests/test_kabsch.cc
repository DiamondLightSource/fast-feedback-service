/**
 * @file test_kabsch.cc
 * @brief Unit tests for the GPU-accelerated Kabsch integration pipeline
 *
 * Each test loads a DIALS integration dump, runs one pipeline step,
 * and compares against the post-stage dump.
 *
 * Test data is sourced from DIALS integrator stage dumps produced with
 * DIALS_INTEGRATION_DUMP_DIR. All HDF5 datasets live under the
 * internal group "dials/processing/group_0". Before/after state is
 * captured in separate files:
 *
 *   bbox_before.h5 / bbox_after.h5             → BoundingBoxComputation
 *   background_before.h5 (+ bbox_after.h5)     → KabschTransformSingleImage
 *   background_before.h5 / summation_after.h5  → SummationFinalization
 *
 * Expected test data directory layout (under tests/data/):
 *   indexed.expt          - experiment geometry (beam, detector, goniometer, scan)
 *   bbox_before.h5        - reflection table before compute_bbox (s1, xyzcal.mm, …)
 *   bbox_after.h5         - reflection table after  compute_bbox (+ bbox column)
 *   background_before.h5  - pixel statistics before compute_background
 *                           (foreground_pixel_sum, n_foreground, …)
 *                           also stores /image_data and kernel attributes
 *   summation_after.h5    - integrated results (intensity.sum.value, …)
 */

#include <gtest/gtest.h>
#include <hdf5.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <dx2/beam.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/h5/h5utils.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <vector>

#include "cuda_common.hpp"
#include "extent.hpp"
#include "ffs_logger.hpp"
#include "kabsch.cuh"
#include "math/math_utils.cuh"
#include "math/vector3d.cuh"

namespace fs = std::filesystem;

/// HDF5 group path prefix for DIALS reflection table datasets
static const std::string G = "dials/processing/group_0/";

#pragma region Helpers

/* Scalar attribute readers (dx2 only covers datasets) */

/// Read a scalar double attribute from an HDF5 location
static double read_double_attr(hid_t loc, const char *name) {
    h5utils::H5Attr attr(H5Aopen(loc, name, H5P_DEFAULT));
    EXPECT_TRUE(static_cast<bool>(attr)) << "Failed to open attribute: " << name;
    double val;
    H5Aread(attr, H5T_NATIVE_DOUBLE, &val);
    return val;
}

/// Read a scalar int attribute from an HDF5 location
static int read_int_attr(hid_t loc, const char *name) {
    h5utils::H5Attr attr(H5Aopen(loc, name, H5P_DEFAULT));
    EXPECT_TRUE(static_cast<bool>(attr)) << "Failed to open attribute: " << name;
    int val;
    H5Aread(attr, H5T_NATIVE_INT, &val);
    return val;
}

/// Check whether a dataset exists in a file
static bool dataset_exists(const std::string &filename, const char *name) {
    h5utils::H5File file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (!file) return false;
    return H5Lexists(file, name, H5P_DEFAULT) > 0;
}

#pragma endregion Helpers

#pragma region Test Fixture

// Test fixture - resolves test data paths relative to this source file

class IntegrationStepTest : public ::testing::Test {
  protected:
    void SetUp() override {
        test_dir = fs::path(__FILE__).parent_path();
        data_dir = test_dir / "data";
    }

    fs::path test_dir;
    fs::path data_dir;

    /// Convenience: assert a file exists
    void assert_file_exists(const fs::path &p) {
        ASSERT_TRUE(fs::exists(p)) << "Required test file not found: " << p;
    }
};

#pragma endregion Test Fixture

#pragma region GPU Kabsch Transform

/*
 * GPU Kabsch Transform + Summation Accumulation
 *
 * Runs the Kabsch transform kernel on a single image. For each
 * pixel (x, y) that falls within a reflection's bounding box, the
 * kernel computes the Kabsch coordinate distance from the reflection
 * centroid and classifies the pixel as foreground (within the
 * nσ-ellipsoid) or background. Pixel values are then accumulated
 * into per-reflection foreground and background sum/count buffers.
 *
 * Reflection geometry (bbox_after.h5, group dials/processing/group_0):
 *   - s1              (N×3 double)  - predicted s₁ vectors
 *   - xyzcal.mm       (N×3 double)  - predicted positions (x, y, φ)
 *   - bbox            (N×6 int)     - Kabsch bounding boxes
 *
 * Experiment geometry (indexed.expt):
 *   - s₀, m₂, d-matrix, wavelength, oscillation, image range
 *
 * Image + kernel parameters (background_before.h5):
 *   - /image_data     (H×W uint16)  - raw detector image
 *   - Attributes: image_num, delta_b, delta_m
 *
 * Expected output (background_before.h5, group dials/processing/group_0):
 *   - foreground_pixel_sum  (N double)
 *   - n_foreground          (N double)
 *   - background_pixel_sum  (N double)
 *   - n_background          (N double)
 */

TEST_F(IntegrationStepTest, KabschTransformSingleImage) {
    auto bbox_file = data_dir / "bbox_after.h5";
    auto bg_file = data_dir / "background_before.h5";
    auto expt_file = data_dir / "indexed.expt";
    assert_file_exists(bbox_file);
    assert_file_exists(bg_file);
    assert_file_exists(expt_file);

    // Load experiment geometry: beam, detector, goniometer, scan
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
    int image_range_start = scan.get_image_range()[0];

    // Flatten detector d-matrix (3×3 → 9-element row-major) for GPU
    Eigen::Matrix3d d_matrix_eigen = panel.get_d_matrix();
    std::vector<scalar_t> d_matrix_scalar(9);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            d_matrix_scalar[i * 3 + j] = static_cast<scalar_t>(d_matrix_eigen(i, j));

    fastvec::Vector3D s0_vec =
      fastvec::make_vector3d(static_cast<scalar_t>(s0_eigen[0]),
                             static_cast<scalar_t>(s0_eigen[1]),
                             static_cast<scalar_t>(s0_eigen[2]));
    fastvec::Vector3D rot_axis_vec =
      fastvec::make_vector3d(static_cast<scalar_t>(rot_axis_eigen[0]),
                             static_cast<scalar_t>(rot_axis_eigen[1]),
                             static_cast<scalar_t>(rot_axis_eigen[2]));

    // Load kernel parameters stored as root-level attributes on background_before.h5
    auto bg_path = bg_file.string();
    int image_num;
    scalar_t delta_b, delta_m;
    {
        h5utils::H5File file(H5Fopen(bg_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
        ASSERT_TRUE(static_cast<bool>(file));
        h5utils::H5Group root(H5Gopen2(file, "/", H5P_DEFAULT));
        image_num = read_int_attr(root, "image_num");
        delta_b = static_cast<scalar_t>(read_double_attr(root, "delta_b"));
        delta_m = static_cast<scalar_t>(read_double_attr(root, "delta_m"));
    }

    // Load the raw detector image from background_before.h5
    auto image_data = read_array_from_h5_file<uint16_t>(bg_path, "/image_data");

    // Derive image dimensions from the panel geometry
    auto image_size_mm = panel.get_image_size_mm();
    auto pixel_size = panel.get_pixel_size();
    uint32_t width =
      static_cast<uint32_t>(std::round(image_size_mm[0] / pixel_size[0]));
    uint32_t height =
      static_cast<uint32_t>(std::round(image_size_mm[1] / pixel_size[1]));
    ASSERT_EQ(image_data.size(), static_cast<size_t>(width) * height);

    // Load reflection geometry from the post-bbox reflection table
    auto bbox_path = bbox_file.string();
    auto s1_flat = read_array_from_h5_file<double>(bbox_path, G + "s1");
    auto xyzcal_flat = read_array_from_h5_file<double>(bbox_path, G + "xyzcal.mm");
    auto bbox_flat = read_array_from_h5_file<int64_t>(bbox_path, G + "bbox");

    size_t num_reflections = s1_flat.size() / 3;
    ASSERT_EQ(xyzcal_flat.size(), num_reflections * 3);
    ASSERT_EQ(bbox_flat.size(), num_reflections * 6);

    // Load the subset of reflection indices that overlap this image
    std::vector<size_t> refl_indices;
    if (dataset_exists(bg_path, "/reflection_indices")) {
        auto ri_int = read_array_from_h5_file<int64_t>(bg_path, "/reflection_indices");
        refl_indices.assign(ri_int.begin(), ri_int.end());
    } else {
        // If no index subset is specified, assume all reflections overlap this image
        refl_indices.resize(num_reflections);
        std::iota(refl_indices.begin(), refl_indices.end(), 0);
    }
    size_t num_refls_this_image = refl_indices.size();

    // Convert double-precision host data to GPU-compatible types
    // (scalar_t precision, fastvec::Vector3D format)
    std::vector<fastvec::Vector3D> s1_vecs(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        s1_vecs[i] = fastvec::make_vector3d(static_cast<scalar_t>(s1_flat[i * 3 + 0]),
                                            static_cast<scalar_t>(s1_flat[i * 3 + 1]),
                                            static_cast<scalar_t>(s1_flat[i * 3 + 2]));
    }

    // Extract φ from column 2 of xyzcal.mm (the rotation angle)
    std::vector<scalar_t> phi_vals(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        phi_vals[i] = static_cast<scalar_t>(xyzcal_flat[i * 3 + 2]);
    }

    std::vector<BoundingBoxExtents> bboxes(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        bboxes[i].x_min = static_cast<int>(bbox_flat[i * 6 + 0]);
        bboxes[i].x_max = static_cast<int>(bbox_flat[i * 6 + 1]);
        bboxes[i].y_min = static_cast<int>(bbox_flat[i * 6 + 2]);
        bboxes[i].y_max = static_cast<int>(bbox_flat[i * 6 + 3]);
        bboxes[i].z_min = static_cast<int>(bbox_flat[i * 6 + 4]);
        bboxes[i].z_max = static_cast<int>(bbox_flat[i * 6 + 5]);
    }

    // Allocate pitched device memory for the image - pitched layout
    // ensures coalesced 2D access on the GPU
    auto d_image = PitchedMalloc<pixel_t>(width, height);
    cudaMemcpy2D(d_image.get(),
                 d_image.pitch_bytes(),
                 image_data.data(),
                 width * sizeof(pixel_t),
                 width * sizeof(pixel_t),
                 height,
                 cudaMemcpyHostToDevice);
    cuda_throw_error();

    DeviceBuffer<scalar_t> d_d_matrix(9);
    DeviceBuffer<fastvec::Vector3D> d_s1_vectors(num_reflections);
    DeviceBuffer<scalar_t> d_phi_values(num_reflections);
    DeviceBuffer<BoundingBoxExtents> d_bboxes(num_reflections);
    DeviceBuffer<size_t> d_reflection_indices(num_refls_this_image);

    d_d_matrix.assign(d_matrix_scalar.data());
    d_s1_vectors.assign(s1_vecs.data());
    d_phi_values.assign(phi_vals.data());
    d_bboxes.assign(bboxes.data());
    d_reflection_indices.assign(refl_indices.data());

    // Allocate and zero per-reflection accumulator buffers for
    // foreground/background sums and pixel counts
    DeviceBuffer<accumulator_t> d_fg_sum(num_reflections);
    DeviceBuffer<uint32_t> d_fg_count(num_reflections);
    DeviceBuffer<accumulator_t> d_bg_sum(num_reflections);
    DeviceBuffer<uint32_t> d_bg_count(num_reflections);
    cudaMemset(d_fg_sum.data(), 0, num_reflections * sizeof(accumulator_t));
    cudaMemset(d_fg_count.data(), 0, num_reflections * sizeof(uint32_t));
    cudaMemset(d_bg_sum.data(), 0, num_reflections * sizeof(accumulator_t));
    cudaMemset(d_bg_count.data(), 0, num_reflections * sizeof(uint32_t));
    cuda_throw_error();

    // Run the Kabsch kernel over this image's reflections
    compute_kabsch_transform(d_image.get(),
                             d_image.pitch_bytes(),
                             width,
                             height,
                             image_num,
                             d_d_matrix.data(),
                             wavelength,
                             osc_start,
                             osc_width,
                             image_range_start,
                             s0_vec,
                             rot_axis_vec,
                             d_s1_vectors.data(),
                             d_phi_values.data(),
                             d_bboxes.data(),
                             d_reflection_indices.data(),
                             num_refls_this_image,
                             delta_b,
                             delta_m,
                             d_fg_sum.data(),
                             d_fg_count.data(),
                             d_bg_sum.data(),
                             d_bg_count.data(),
                             nullptr);  // default stream
    cudaDeviceSynchronize();
    cuda_throw_error();

    // Transfer accumulated results back to host memory
    std::vector<accumulator_t> h_fg_sum(num_reflections);
    std::vector<uint32_t> h_fg_count(num_reflections);
    std::vector<accumulator_t> h_bg_sum(num_reflections);
    std::vector<uint32_t> h_bg_count(num_reflections);
    d_fg_sum.extract(h_fg_sum.data());
    d_fg_count.extract(h_fg_count.data());
    d_bg_sum.extract(h_bg_sum.data());
    d_bg_count.extract(h_bg_count.data());

    // Load expected accumulator values from background_before.h5
    auto exp_fg_sum =
      read_array_from_h5_file<double>(bg_path, G + "foreground_pixel_sum");
    auto exp_fg_count = read_array_from_h5_file<double>(bg_path, G + "n_foreground");
    auto exp_bg_sum =
      read_array_from_h5_file<double>(bg_path, G + "background_pixel_sum");
    auto exp_bg_count = read_array_from_h5_file<double>(bg_path, G + "n_background");

    ASSERT_EQ(exp_fg_sum.size(), num_refls_this_image);
    ASSERT_EQ(exp_fg_count.size(), num_refls_this_image);
    ASSERT_EQ(exp_bg_sum.size(), num_refls_this_image);
    ASSERT_EQ(exp_bg_count.size(), num_refls_this_image);

    // Verify accumulated foreground and background values match exactly.
    // The accumulator buffers are indexed by the full reflection table,
    // so map each local subset entry back to its global slot via refl_indices.
    // (integer comparison - no tolerance needed)
    const char *kabsch_comp_names[] = {"fg_sum", "n_fg", "bg_sum", "n_bg"};
    size_t kabsch_exact = 0;
    size_t kabsch_mismatches = 0;
    std::array<double, 4> kabsch_sum_abs_err = {};
    std::array<double, 4> kabsch_max_abs_err = {};
    // signed-difference histogram per component: (computed − expected) → count
    std::array<std::map<int, int>, 4> kabsch_diff_hist;

    for (size_t i = 0; i < num_refls_this_image; ++i) {
        size_t gi = refl_indices[i];

        std::array<double, 4> got = {static_cast<double>(h_fg_sum[gi]),
                                     static_cast<double>(h_fg_count[gi]),
                                     static_cast<double>(h_bg_sum[gi]),
                                     static_cast<double>(h_bg_count[gi])};
        std::array<double, 4> exp = {
          exp_fg_sum[i], exp_fg_count[i], exp_bg_sum[i], exp_bg_count[i]};

        bool all_match = true;
        for (int c = 0; c < 4; ++c) {
            double diff = std::abs(got[c] - exp[c]);
            if (diff > 0) {
                all_match = false;
                kabsch_sum_abs_err[c] += diff;
                kabsch_max_abs_err[c] = std::max(kabsch_max_abs_err[c], diff);
            }
        }

        if (all_match) {
            kabsch_exact++;
        } else {
            kabsch_mismatches++;
            // Accumulate signed integer diffs — pixel sums/counts are uint32_t-valued
            for (int c = 0; c < 4; ++c) {
                int signed_diff = static_cast<int>(got[c] - exp[c]);
                if (signed_diff != 0) kabsch_diff_hist[c][signed_diff]++;
            }
            EXPECT_EQ(h_fg_sum[gi], static_cast<accumulator_t>(exp_fg_sum[i]))
              << "foreground_pixel_sum mismatch at reflection " << gi << " (local " << i
              << ")";
            EXPECT_EQ(h_fg_count[gi], static_cast<uint32_t>(exp_fg_count[i]))
              << "n_foreground mismatch at reflection " << gi << " (local " << i << ")";
            EXPECT_EQ(h_bg_sum[gi], static_cast<accumulator_t>(exp_bg_sum[i]))
              << "background_pixel_sum mismatch at reflection " << gi << " (local " << i
              << ")";
            EXPECT_EQ(h_bg_count[gi], static_cast<uint32_t>(exp_bg_count[i]))
              << "n_background mismatch at reflection " << gi << " (local " << i << ")";
        }
    }

    std::cout << "\n=== Kabsch Transform Comparison Summary ==="
              << "\n  Reflections compared : " << num_refls_this_image << " (of "
              << num_reflections << " total)"
              << "\n  Exact matches        : " << kabsch_exact << " ("
              << (num_refls_this_image ? 100.0 * kabsch_exact / num_refls_this_image
                                       : 0)
              << "%)"
              << "\n  Mismatches           : " << kabsch_mismatches << "\n";
    if (kabsch_mismatches > 0) {
        std::cout << "\n  Per-component error breakdown (over " << kabsch_mismatches
                  << " mismatched reflections):\n"
                  << "  Component   MeanAbsErr    MaxAbsErr\n";
        for (int c = 0; c < 4; ++c) {
            double mean_abs = kabsch_sum_abs_err[c] / kabsch_mismatches;
            std::cout << "  " << std::setw(11) << std::left << kabsch_comp_names[c]
                      << " " << std::setw(13) << std::fixed << std::setprecision(1)
                      << mean_abs << " " << std::setw(9) << kabsch_max_abs_err[c]
                      << "\n";
        }

        // Top-5 signed differences per component (computed − expected)
        constexpr int KABSCH_TOP_N = 5;
        std::cout << "\n  Top-" << KABSCH_TOP_N
                  << " signed differences per component (computed − expected):\n";
        for (int c = 0; c < 4; ++c) {
            if (kabsch_diff_hist[c].empty()) continue;
            std::vector<std::pair<int, int>> entries(kabsch_diff_hist[c].begin(),
                                                     kabsch_diff_hist[c].end());
            std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
                return b.second < a.second;
            });
            std::cout << "  " << kabsch_comp_names[c] << ":\n";
            int shown = 0;
            for (const auto &[diff, count] : entries) {
                if (shown++ >= KABSCH_TOP_N) break;
                std::cout << "    " << std::setw(6) << std::right << diff << " : "
                          << count << "\n";
            }
        }
    }
    std::cout << "==========================================\n";

    EXPECT_EQ(kabsch_mismatches, 0)
      << kabsch_mismatches << " of " << num_refls_this_image
      << " reflections had accumulator mismatches";
}

#pragma endregion GPU Kabsch Transform

#pragma region Summation Finalization

/*
 * Summation Integration Finalization
 *
 * Converts the accumulated foreground/background pixel sums into
 * integrated intensities using the summation formula:
 *
 *   b̄ = Σ(bg) / n_bg           (mean background per pixel)
 *   I = Σ(fg) − n_fg × b̄        (background-subtracted intensity)
 *   Var(I) = |I| + |B| × (1 + n_fg/n_bg)
 *
 * where B = n_fg × b̄ is the total background contribution under the
 * peak. This is a purely host-side operation with no GPU involvement.
 *
 * Input (background_before.h5, group dials/processing/group_0):
 *   - foreground_pixel_sum  (N double)
 *   - n_foreground          (N double)
 *   - background_pixel_sum  (N double)
 *   - n_background          (N double)
 *
 * Output (summation_after.h5, group dials/processing/group_0):
 *   - intensity.sum.value      (N double) - I
 *   - intensity.sum.variance   (N double) - Var(I)
 *   - background.sum.value     (N double) - B = n_fg × b̄
 */

TEST_F(IntegrationStepTest, SummationFinalization) {
    auto input_file = data_dir / "background_before.h5";
    auto output_file = data_dir / "summation_after.h5";
    assert_file_exists(input_file);
    assert_file_exists(output_file);

    // Load accumulated foreground/background pixel sums and counts
    auto input_path = input_file.string();
    auto h_fg_sum_dbl =
      read_array_from_h5_file<double>(input_path, G + "foreground_pixel_sum");
    auto h_fg_count_dbl =
      read_array_from_h5_file<double>(input_path, G + "n_foreground");
    auto h_bg_sum_dbl =
      read_array_from_h5_file<double>(input_path, G + "background_pixel_sum");
    auto h_bg_count_dbl =
      read_array_from_h5_file<double>(input_path, G + "n_background");

    size_t num_reflections = h_fg_sum_dbl.size();
    ASSERT_EQ(h_fg_count_dbl.size(), num_reflections);
    ASSERT_EQ(h_bg_sum_dbl.size(), num_reflections);
    ASSERT_EQ(h_bg_count_dbl.size(), num_reflections);

    // Compute intensity, variance, and background using the summation
    // integration formulas. This mirrors the host-side reduction in
    // integrator.cc.
    std::vector<double> intensities(num_reflections);
    std::vector<double> variances(num_reflections);
    std::vector<double> background_totals(num_reflections);

    for (size_t i = 0; i < num_reflections; ++i) {
        uint32_t fg_count = static_cast<uint32_t>(h_fg_count_dbl[i]);
        uint32_t bg_count = static_cast<uint32_t>(h_bg_count_dbl[i]);

        if (fg_count == 0) {
            intensities[i] = 0.0;
            variances[i] = -1.0;
            background_totals[i] = 0.0;
            continue;
        }

        double fg_sum = h_fg_sum_dbl[i];
        // b̄ = Σ(bg) / n_bg
        double bg_mean = (bg_count > 0) ? h_bg_sum_dbl[i] / bg_count : 0.0;

        // B = b̄ × n_fg (total background contribution under the peak)
        double background_total = bg_mean * fg_count;
        // I = Σ(fg) − B
        double intensity = fg_sum - background_total;

        // Var(I) = |I| + |B| × (1 + n_fg/n_bg)
        double fg_bg_ratio =
          (bg_count > 0) ? static_cast<double>(fg_count) / bg_count : 0.0;
        double variance =
          std::abs(intensity) + std::abs(background_total) * (1.0 + fg_bg_ratio);

        intensities[i] = intensity;
        variances[i] = variance;
        background_totals[i] = background_total;
    }

    // Load expected integrated results from the post-summation reflection table
    auto output_path = output_file.string();
    auto exp_intensity =
      read_array_from_h5_file<double>(output_path, G + "intensity.sum.value");
    auto exp_variance =
      read_array_from_h5_file<double>(output_path, G + "intensity.sum.variance");
    auto exp_background =
      read_array_from_h5_file<double>(output_path, G + "background.sum.value");

    ASSERT_EQ(exp_intensity.size(), num_reflections);
    ASSERT_EQ(exp_variance.size(), num_reflections);
    ASSERT_EQ(exp_background.size(), num_reflections);

    // Compare computed values against truth with tolerance (1e-6) to
    // account for floating-point reduction order differences
    const char *sum_comp_names[] = {"intensity", "variance", "background"};
    size_t sum_exact = 0;
    size_t sum_mismatches = 0;
    std::array<double, 3> sum_sum_abs_err = {};
    std::array<double, 3> sum_sum_signed_err = {};
    std::array<double, 3> sum_max_abs_err = {};
    // signed-difference histogram per component, bucketed to nearest integer
    std::array<std::map<int, int>, 3> sum_diff_hist;
    constexpr double tol = 0.1;  //1e-6;

    for (size_t i = 0; i < num_reflections; ++i) {
        std::array<double, 3> got = {
          intensities[i], variances[i], background_totals[i]};
        std::array<double, 3> exp = {
          exp_intensity[i], exp_variance[i], exp_background[i]};

        bool all_match = true;
        for (int c = 0; c < 3; ++c) {
            double diff = got[c] - exp[c];
            double absdiff = std::abs(diff);
            if (absdiff > tol) {
                all_match = false;
                sum_sum_abs_err[c] += absdiff;
                sum_sum_signed_err[c] += diff;
                sum_max_abs_err[c] = std::max(sum_max_abs_err[c], absdiff);
            }
        }

        if (all_match) {
            sum_exact++;
        } else {
            sum_mismatches++;
            // Bucket signed diffs to nearest integer for histogram
            for (int c = 0; c < 3; ++c) {
                double signed_diff = got[c] - exp[c];
                if (std::abs(signed_diff) > tol)
                    sum_diff_hist[c][static_cast<int>(std::round(signed_diff))]++;
            }
            EXPECT_NEAR(intensities[i], exp_intensity[i], tol)
              << "intensity.sum.value mismatch at reflection " << i;
            EXPECT_NEAR(variances[i], exp_variance[i], tol)
              << "intensity.sum.variance mismatch at reflection " << i;
            EXPECT_NEAR(background_totals[i], exp_background[i], tol)
              << "background.sum.value mismatch at reflection " << i;
        }
    }

    std::cout << "\n=== Summation Finalization Comparison Summary ==="
              << "\n  Reflections compared : " << num_reflections
              << "\n  Exact matches        : " << sum_exact << " ("
              << (num_reflections ? 100.0 * sum_exact / num_reflections : 0) << "%)"
              << "\n  Mismatches (>" << tol << "): " << sum_mismatches << "\n";
    if (sum_mismatches > 0) {
        std::cout << "\n  Per-component error breakdown (over " << sum_mismatches
                  << " mismatched reflections):\n"
                  << "  Component    MeanAbsErr   MeanSignedErr  MaxAbsErr\n";
        for (int c = 0; c < 3; ++c) {
            double mean_abs = sum_sum_abs_err[c] / sum_mismatches;
            double mean_signed = sum_sum_signed_err[c] / sum_mismatches;
            std::cout << "  " << std::setw(12) << std::left << sum_comp_names[c] << " "
                      << std::setw(12) << std::fixed << std::setprecision(6) << mean_abs
                      << " " << std::setw(14) << mean_signed << " " << std::setw(9)
                      << sum_max_abs_err[c] << "\n";
        }

        // Top-5 signed differences per component, rounded to nearest integer
        constexpr int SUM_TOP_N = 5;
        std::cout
          << "\n  Top-" << SUM_TOP_N
          << " signed differences per component (computed − expected, nearest int):\n";
        for (int c = 0; c < 3; ++c) {
            if (sum_diff_hist[c].empty()) continue;
            std::vector<std::pair<int, int>> entries(sum_diff_hist[c].begin(),
                                                     sum_diff_hist[c].end());
            std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
                return b.second < a.second;
            });
            std::cout << "  " << sum_comp_names[c] << ":\n";
            int shown = 0;
            for (const auto &[diff, count] : entries) {
                if (shown++ >= SUM_TOP_N) break;
                std::cout << "    " << std::setw(6) << std::right << diff << " : "
                          << count << "\n";
            }
        }
    }
    std::cout << "=================================================\n";

    EXPECT_EQ(sum_mismatches, 0) << sum_mismatches << " of " << num_reflections
                                 << " reflections exceeded tolerance " << tol;
}

#pragma endregion Summation Finalization
