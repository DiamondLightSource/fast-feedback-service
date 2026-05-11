/**
 * @file test_extent.cc
 * @brief Unit tests for compute_kabsch_bounding_boxes (CPU, extent.cc)
 *
 * Tests the Kabsch bounding box computation against DIALS integrator
 * stage dumps. Input reflection geometry comes from bbox_before.h5
 * (pre-compute_bbox), and expected bounding boxes from bbox_after.h5
 * (post-compute_bbox). All HDF5 datasets are under the internal group
 * "dials/processing/group_0".
 */

#include <gtest/gtest.h>
#include <hdf5.h>

#include <Eigen/Dense>
#include <algorithm>
#include <dx2/beam.hpp>
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

#include "ffs_logger.hpp"
#include "integrator/extent.hpp"
#include "math/math_utils.cuh"

namespace fs = std::filesystem;

/// HDF5 group path prefix for DIALS reflection table datasets
static const std::string G = "dials/processing/group_0/";

class ExtentTest : public ::testing::Test {
  protected:
    void SetUp() override {
        test_dir = fs::path(__FILE__).parent_path();
        data_dir = test_dir / "data";
    }

    fs::path test_dir;
    fs::path data_dir;

    void assert_file_exists(const fs::path &p) {
        ASSERT_TRUE(fs::exists(p)) << "Required test file not found: " << p;
    }
};

/*
 * Bounding Box Computation
 *
 * Computes Kabsch integration bounding boxes for each reflection.
 * The detector-plane extents (x, y) are determined by projecting
 * the beam divergence envelope (±Δb in Kabsch e₁/e₂ directions)
 * back onto the Ewald sphere and then onto the detector via
 * ray-intersection. The rotation-axis extent (z) is derived from
 * the mosaicity parameter: φ′ = φᶜ ± Δm/ζ, where ζ = m₂ · e₁.
 *
 * Input (bbox_before.h5, group dials/processing/group_0):
 *   - s1            (N×3 double) - predicted s₁ vectors
 *   - xyzcal.mm     (N×3 double) - predicted positions (x, y, φ)
 *   - Attributes on root: σ_b, σ_m (radians)
 *
 * Geometry (indexed.expt):
 *   - s₀ from MonochromaticBeam, m₂ from Goniometer
 *
 * Output (bbox_after.h5, group dials/processing/group_0):
 *   - bbox          (N×6 int)    - [x_min, x_max, y_min, y_max, z_min, z_max]
 */

TEST_F(ExtentTest, ComputeKabschBoundingBoxes) {
    auto before_file = data_dir / "bbox_before.h5";
    auto after_file = data_dir / "bbox_after.h5";
    auto expt_file = data_dir / "indexed.expt";
    assert_file_exists(before_file);
    assert_file_exists(after_file);
    assert_file_exists(expt_file);

    // Helper: print the actual HDF5 type size for a dataset
    auto print_h5_type_size = [](const std::string &filepath,
                                 const std::string &dataset_path) {
        hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) {
            std::cout << "  [could not open file]\n";
            return;
        }
        hid_t dset = H5Dopen(file, dataset_path.c_str(), H5P_DEFAULT);
        if (dset < 0) {
            H5Fclose(file);
            std::cout << "  [could not open dataset]\n";
            return;
        }
        hid_t dtype = H5Dget_type(dset);
        std::cout << "  HDF5 type size = " << H5Tget_size(dtype) << " bytes"
                  << ", class = " << H5Tget_class(dtype) << "\n";
        H5Tclose(dtype);
        H5Dclose(dset);
        H5Fclose(file);
    };

    // Load predicted s₁ vectors and calculated positions from the pre-bbox reflection table
    auto before_path = before_file.string();

    // std::cout << "Reading " << G + "s1" << " as double (sizeof=" << sizeof(double) << ")...\n";
    // print_h5_type_size(before_path, G + "s1");
    auto s1_flat = read_array_from_h5_file<double>(before_path, G + "s1");
    // std::cout << "  -> read " << s1_flat.size() << " elements\n";

    // std::cout << "Reading " << G + "xyzcal.mm" << " as double (sizeof=" << sizeof(double) << ")...\n";
    // print_h5_type_size(before_path, G + "xyzcal.mm");
    auto xyzcal_flat = read_array_from_h5_file<double>(before_path, G + "xyzcal.mm");
    // std::cout << "  -> read " << xyzcal_flat.size() << " elements\n";

    // σ_b (beam divergence) and σ_m (mosaicity) - fixed test values
    // double sigma_b = degrees_to_radians(0.03);
    // double sigma_m = degrees_to_radians(0.03);

    size_t num_reflections = s1_flat.size() / 3;
    ASSERT_EQ(xyzcal_flat.size(), num_reflections * 3);

    // Wrap flat arrays as 2D mdspan views for compute_kabsch_bounding_boxes
    using mdspan_2d_double =
      std::experimental::mdspan<double, std::experimental::dextents<size_t, 2>>;
    mdspan_2d_double s1_vectors(s1_flat.data(), num_reflections, 3);
    mdspan_2d_double phi_column(xyzcal_flat.data(), num_reflections, 3);

    // Extract detector panel, scan, and beam parameters from the experiment
    std::ifstream f(expt_file);
    auto elist_json = nlohmann::json::parse(f);
    Experiment<MonochromaticBeam> expt(elist_json);
    const Panel &panel = expt.detector().panels()[0];
    const Scan &scan = expt.scan();
    MonochromaticBeam beam = expt.beam();

    // σ_b (beam divergence) and σ_m (mosaicity) from the experiment JSON
    double sigma_b =
      degrees_to_radians(elist_json["profile"][0]["sigma_b"].get<double>());
    double sigma_m =
      degrees_to_radians(elist_json["profile"][0]["sigma_m"].get<double>());

    // s₀ and rotation axis come from the experiment geometry
    Eigen::Vector3d s0 = beam.get_s0();
    Eigen::Vector3d rotation_axis = expt.goniometer().get_rotation_axis();

    // Compute Kabsch bounding boxes using σ_b, σ_m and the Ewald sphere geometry
    auto computed_bboxes = compute_kabsch_bounding_boxes(s0,
                                                         rotation_axis,
                                                         s1_vectors,
                                                         phi_column,
                                                         num_reflections,
                                                         sigma_b,
                                                         sigma_m,
                                                         panel,
                                                         scan,
                                                         beam);

    ASSERT_EQ(computed_bboxes.size(), num_reflections);

    // Load Miller indices (h, k, l) for matching computed bboxes to expected
    auto before_hkl_flat =
      read_array_from_h5_file<int64_t>(before_path, G + "miller_index");
    ASSERT_EQ(before_hkl_flat.size(), num_reflections * 3);

    // Load expected bounding boxes and their Miller indices from the post-bbox file
    auto after_path = after_file.string();
    auto expected_bbox = read_array_from_h5_file<int64_t>(after_path, G + "bbox");
    auto after_hkl_flat =
      read_array_from_h5_file<int64_t>(after_path, G + "miller_index");
    size_t num_expected = after_hkl_flat.size() / 3;
    ASSERT_EQ(expected_bbox.size(), num_expected * 6);

    // (h,k,l) → row index in the after file; multimap because the
    // same HKL can appear on adjacent images
    using HKL = std::tuple<int64_t, int64_t, int64_t>;
    std::multimap<HKL, size_t> after_hkl_to_row;
    for (size_t i = 0; i < num_expected; ++i) {
        HKL key{after_hkl_flat[i * 3 + 0],
                after_hkl_flat[i * 3 + 1],
                after_hkl_flat[i * 3 + 2]};
        after_hkl_to_row.emplace(key, i);
    }

    // Match computed bboxes to expected by HKL; consume first exact
    // match to avoid ordering dependence
    size_t mismatches = 0;
    size_t not_found = 0;
    size_t exact_matches = 0;
    size_t compared = 0;

    // Per-component error accumulators: [x_min, x_max, y_min, y_max, z_min, z_max]
    const char *comp_names[] = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"};
    std::array<double, 6> sum_abs_err = {};  // Σ|computed − expected|
    std::array<double, 6> sum_signed_err =
      {};  // Σ(computed − expected) - detect systematic bias
    std::array<int, 6> max_abs_err = {};    // max |computed − expected|
    std::array<size_t, 6> off_by_one = {};  // count of |error| == 1 (rounding boundary)
    // signed-difference histogram per component: diff → count
    std::array<std::map<int, int>, 6> diff_hist;

    for (size_t i = 0; i < num_reflections; ++i) {
        HKL key{before_hkl_flat[i * 3 + 0],
                before_hkl_flat[i * 3 + 1],
                before_hkl_flat[i * 3 + 2]};

        auto [it, end] = after_hkl_to_row.equal_range(key);
        if (it == end) {
            not_found++;
            ADD_FAILURE() << "Reflection " << i << " with HKL (" << std::get<0>(key)
                          << ", " << std::get<1>(key) << ", " << std::get<2>(key)
                          << ") not found in expected file";
            continue;
        }

        const auto &bbox = computed_bboxes[i];
        std::array<int, 6> got = {
          bbox.x_min, bbox.x_max, bbox.y_min, bbox.y_max, bbox.z_min, bbox.z_max};

        // Search for an exact match among all after-file rows sharing this HKL
        auto match_it = end;
        for (auto candidate = it; candidate != end; ++candidate) {
            size_t j = candidate->second;
            bool equal = true;
            for (int c = 0; c < 6; ++c)
                equal &= (got[c] == static_cast<int>(expected_bbox[j * 6 + c]));
            if (equal) {
                match_it = candidate;
                break;
            }
        }

        if (match_it != end) {
            // Consume the matched row so it cannot be reused by another reflection
            after_hkl_to_row.erase(match_it);
            exact_matches++;
        } else {
            // No exact match - accumulate error statistics against the first candidate
            mismatches++;
            size_t j = it->second;
            std::array<int, 6> exp;
            for (int c = 0; c < 6; ++c)
                exp[c] = static_cast<int>(expected_bbox[j * 6 + c]);

            for (int c = 0; c < 6; ++c) {
                int diff = got[c] - exp[c];
                int absdiff = std::abs(diff);
                sum_abs_err[c] += absdiff;
                sum_signed_err[c] += diff;
                max_abs_err[c] = std::max(max_abs_err[c], absdiff);
                if (absdiff == 1) off_by_one[c]++;
                diff_hist[c][diff]++;
            }

            auto hkl_str = "(" + std::to_string(std::get<0>(key)) + ", "
                           + std::to_string(std::get<1>(key)) + ", "
                           + std::to_string(std::get<2>(key)) + ")";
            EXPECT_EQ(bbox.x_min, exp[0]) << "x_min mismatch at HKL " << hkl_str;
            EXPECT_EQ(bbox.x_max, exp[1]) << "x_max mismatch at HKL " << hkl_str;
            EXPECT_EQ(bbox.y_min, exp[2]) << "y_min mismatch at HKL " << hkl_str;
            EXPECT_EQ(bbox.y_max, exp[3]) << "y_max mismatch at HKL " << hkl_str;
            EXPECT_EQ(bbox.z_min, exp[4]) << "z_min mismatch at HKL " << hkl_str;
            EXPECT_EQ(bbox.z_max, exp[5]) << "z_max mismatch at HKL " << hkl_str;
        }
        compared++;
    }

    // Summary statistics
    std::cout << "\n=== Bounding Box Comparison Summary ===\n";
    std::cout << "  Reflections compared : " << compared << "\n";
    std::cout << "  Exact matches        : " << exact_matches << " ("
              << (compared ? 100.0 * exact_matches / compared : 0) << "%)\n";
    std::cout << "  Mismatches           : " << mismatches << "\n";
    std::cout << "  Not found in expected: " << not_found << "\n";

    if (mismatches > 0) {
        std::cout << "\n  Per-component error breakdown (over " << mismatches
                  << " mismatched reflections):\n";
        std::cout << "  Component  MeanAbsErr  MeanSignedErr  MaxAbsErr  OffByOne\n";
        for (int c = 0; c < 6; ++c) {
            double mean_abs = sum_abs_err[c] / mismatches;
            double mean_signed = sum_signed_err[c] / mismatches;
            std::cout << "  " << std::setw(9) << std::left << comp_names[c] << "  "
                      << std::setw(10) << std::fixed << std::setprecision(3) << mean_abs
                      << "  " << std::setw(13) << std::fixed << std::setprecision(3)
                      << mean_signed << "  " << std::setw(9) << max_abs_err[c] << "  "
                      << off_by_one[c] << "\n";
        }

        // Signed-difference histograms: top-5 most frequent
        // values per component, sorted by descending frequency.
        constexpr int TOP_N = 5;
        std::cout << "\n  Top-" << TOP_N
                  << " signed differences per component (computed - expected):\n";
        for (int c = 0; c < 6; ++c) {
            if (diff_hist[c].empty()) continue;
            // Collect and sort by descending count
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
    std::cout << "=======================================\n";

    EXPECT_EQ(not_found, 0) << not_found
                            << " reflections had no HKL match in expected file";
    EXPECT_EQ(mismatches, 0) << mismatches << " of " << num_reflections
                             << " bounding boxes mismatched";
}
