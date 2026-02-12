#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <dx2/beam.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "extent.hpp"
#include "ffs_logger.hpp"
#include "math/math_utils.cuh"

/**
 * @brief Validate computed bounding boxes against existing bbox column
 * 
 * Compares the first N bounding boxes and writes results to a validation file.
 * 
 * @param reflections The reflection table containing existing bbox column
 * @param computed_bbox_data Flat array of computed bounding boxes (6 values per reflection)
 * @param num_reflections Total number of reflections
 * @param num_to_display Number of bounding boxes to display for comparison
 * @return 0 on success, 1 on error
 */
int validate_bounding_boxes(ReflectionTable &reflections,
                            const std::vector<double> &computed_bbox_data,
                            size_t num_reflections,
                            size_t num_to_display = 10,
                            const std::string &data_path = "") {
    logger.info(
      "Validation mode: comparing computed bounding boxes with existing bbox column");

    // Load existing bounding boxes
    auto bbox_column_opt = reflections.column<int>(data_path + "bbox");
    if (!bbox_column_opt) {
        logger.error("Column 'bbox' not found in reflection data for validation.");
        return 1;
    }
    auto bbox_column = *bbox_column_opt;

    // Statistics for comparison
    double total_x_min_diff = 0.0, total_x_max_diff = 0.0;
    double total_y_min_diff = 0.0, total_y_max_diff = 0.0;
    double total_z_min_diff = 0.0, total_z_max_diff = 0.0;
    double max_x_min_diff = 0.0, max_x_max_diff = 0.0;
    double max_y_min_diff = 0.0, max_y_max_diff = 0.0;
    double max_z_min_diff = 0.0, max_z_max_diff = 0.0;

    logger.info("First {} bounding box comparisons:", num_to_display);
    for (size_t i = 0; i < num_reflections; ++i) {
        // Existing bbox
        double ex_x_min = bbox_column(i, 0);
        double ex_x_max = bbox_column(i, 1);
        double ex_y_min = bbox_column(i, 2);
        double ex_y_max = bbox_column(i, 3);
        int ex_z_min = static_cast<int>(bbox_column(i, 4));
        int ex_z_max = static_cast<int>(bbox_column(i, 5));

        // Computed bbox (from flat array)
        double comp_x_min = computed_bbox_data[i * 6 + 0];
        double comp_x_max = computed_bbox_data[i * 6 + 1];
        double comp_y_min = computed_bbox_data[i * 6 + 2];
        double comp_y_max = computed_bbox_data[i * 6 + 3];
        int comp_z_min = static_cast<int>(computed_bbox_data[i * 6 + 4]);
        int comp_z_max = static_cast<int>(computed_bbox_data[i * 6 + 5]);

        // Compute differences
        double diff_x_min = std::abs(comp_x_min - ex_x_min);
        double diff_x_max = std::abs(comp_x_max - ex_x_max);
        double diff_y_min = std::abs(comp_y_min - ex_y_min);
        double diff_y_max = std::abs(comp_y_max - ex_y_max);
        double diff_z_min = std::abs(comp_z_min - ex_z_min);
        double diff_z_max = std::abs(comp_z_max - ex_z_max);

        // Accumulate statistics
        total_x_min_diff += diff_x_min;
        total_x_max_diff += diff_x_max;
        total_y_min_diff += diff_y_min;
        total_y_max_diff += diff_y_max;
        total_z_min_diff += diff_z_min;
        total_z_max_diff += diff_z_max;

        max_x_min_diff = std::max(max_x_min_diff, diff_x_min);
        max_x_max_diff = std::max(max_x_max_diff, diff_x_max);
        max_y_min_diff = std::max(max_y_min_diff, diff_y_min);
        max_y_max_diff = std::max(max_y_max_diff, diff_y_max);
        max_z_min_diff = std::max(max_z_min_diff, diff_z_min);
        max_z_max_diff = std::max(max_z_max_diff, diff_z_max);

        // Display first N comparisons
        if (i < num_to_display) {
            logger.info(
              "bbox[{}]: existing x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] z=[{},{}]",
              i,
              ex_x_min,
              ex_x_max,
              ex_y_min,
              ex_y_max,
              ex_z_min,
              ex_z_max);
            logger.info(
              "bbox[{}]: computed x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] z=[{},{}]",
              i,
              comp_x_min,
              comp_x_max,
              comp_y_min,
              comp_y_max,
              comp_z_min,
              comp_z_max);
            logger.info(
              "bbox[{}]: diff     x=[{:.1f},{:.1f}] y=[{:.1f},{:.1f}] "
              "z=[{:.1f},{:.1f}]",
              i,
              diff_x_min,
              diff_x_max,
              diff_y_min,
              diff_y_max,
              diff_z_min,
              diff_z_max);
        }
    }

    // Compute and display average differences
    double n = static_cast<double>(num_reflections);
    logger.info("\n=== Bounding Box Comparison Statistics ===");
    logger.info("Mean absolute differences:");
    logger.info("  x_min: {:.3f} pixels, x_max: {:.3f} pixels",
                total_x_min_diff / n,
                total_x_max_diff / n);
    logger.info("  y_min: {:.3f} pixels, y_max: {:.3f} pixels",
                total_y_min_diff / n,
                total_y_max_diff / n);
    logger.info("  z_min: {:.3f} frames,  z_max: {:.3f} frames",
                total_z_min_diff / n,
                total_z_max_diff / n);
    logger.info("Maximum absolute differences:");
    logger.info(
      "  x_min: {:.3f} pixels, x_max: {:.3f} pixels", max_x_min_diff, max_x_max_diff);
    logger.info(
      "  y_min: {:.3f} pixels, y_max: {:.3f} pixels", max_y_min_diff, max_y_max_diff);
    logger.info(
      "  z_min: {:.3f} frames,  z_max: {:.3f} frames", max_z_min_diff, max_z_max_diff);

    // Add computed bounding boxes to reflection table for comparison
    reflections.add_column(
      "computed_bbox", std::vector<size_t>{num_reflections, 6}, computed_bbox_data);

    // Write output file with comparison
    reflections.write("validation_reflections.h5");
    logger.info("Validation results saved to validation_reflections.h5");

    return 0;
}

// Test fixture for integrator tests with file paths
class ExtentValidationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Get paths relative to test directory
        test_dir = std::filesystem::path(__FILE__).parent_path();
        predicted_refl = test_dir / "data" / "predicted.refl";
        indexed_expt = test_dir / "data" / "indexed.expt";
        integrated_truth_refl = test_dir / "data" / "simple_integrated.refl";
    }

    std::filesystem::path test_dir;
    std::filesystem::path predicted_refl;
    std::filesystem::path indexed_expt;
    std::filesystem::path integrated_truth_refl;
};

TEST_F(ExtentValidationTest, ComputeKabschBoundingBoxes) {
    // Check if test files exist
    ASSERT_TRUE(std::filesystem::exists(predicted_refl))
      << "Reflection file not found: " << predicted_refl;
    ASSERT_TRUE(std::filesystem::exists(indexed_expt))
      << "Experiment file not found: " << indexed_expt;
    ASSERT_TRUE(std::filesystem::exists(integrated_truth_refl))
      << "Integrated reflection file not found: " << integrated_truth_refl;

    // Load source reflection data (indexed reflections for computing)
    logger.info("Loading indexed data from file: {}", predicted_refl.string());
    ReflectionTable reflections(predicted_refl.string());

    // Load integrated reflections for comparison (contains bbox)
    logger.info("Loading integrated data for comparison from: {}",
                integrated_truth_refl.string());
    ReflectionTable integrated_reflections(integrated_truth_refl.string());

    // Log available columns in integrated reflections
    // logger.info("Available columns in integrated reflections:");
    // for (const auto &name : integrated_reflections.get_column_names()) {
    //     logger.info("  - {}", name);
    // }

    // HDF5 data path prefixes for reflection columns
    const std::string indexed_data_path = "";
    const std::string integrated_data_path = "";

    // Extract required columns from indexed reflections
    auto s1_vectors_opt = reflections.column<double>(indexed_data_path + "s1");
    ASSERT_TRUE(s1_vectors_opt.has_value())
      << "Column 's1' not found in reflection data.";
    auto s1_vectors = *s1_vectors_opt;

    auto phi_column_opt = reflections.column<double>(indexed_data_path + "xyzcal.mm");
    ASSERT_TRUE(phi_column_opt.has_value())
      << "Column 'xyzcal.mm' not found for phi positions.";
    auto phi_column = *phi_column_opt;

    // Extract bbox from integrated reflections for comparison
    auto bbox_column_opt =
      integrated_reflections.column<int>(integrated_data_path + "bbox");
    ASSERT_TRUE(bbox_column_opt.has_value())
      << "Column 'bbox' not found in integrated reflection data.";
    auto bbox_column = *bbox_column_opt;

    size_t num_reflections = s1_vectors.extent(0);
    logger.info("Processing {} reflections", num_reflections);

    // Parse experiment list from JSON
    std::ifstream f(indexed_expt);
    nlohmann::json elist_json_obj;
    try {
        elist_json_obj = nlohmann::json::parse(f);
    } catch (nlohmann::json::parse_error &ex) {
        FAIL() << "Failed to parse experiment file: " << ex.what();
    }

    // Construct Experiment object
    Experiment<MonochromaticBeam> expt;
    try {
        expt = Experiment<MonochromaticBeam>(elist_json_obj);
    } catch (const std::invalid_argument &ex) {
        FAIL() << "Failed to construct Experiment: " << ex.what();
    }

    // Extract experimental components
    MonochromaticBeam beam = expt.beam();
    Goniometer gonio = expt.goniometer();
    const Panel &panel = expt.detector().panels()[0];
    const Scan &scan = expt.scan();

    Eigen::Vector3d s0 = beam.get_s0();
    Eigen::Vector3d rotation_axis = gonio.get_rotation_axis();

    // Use baseline values for sigma_b and sigma_m
    double sigma_b = degrees_to_radians(0.01);
    double sigma_m = degrees_to_radians(0.01);

    // Compute bounding boxes using CPU-based Kabsch coordinate system
    logger.info("Computing Kabsch bounding boxes for {} reflections", num_reflections);

    std::vector<BoundingBoxExtents> computed_bboxes =
      compute_kabsch_bounding_boxes(s0,
                                    rotation_axis,
                                    s1_vectors,
                                    phi_column,
                                    num_reflections,
                                    sigma_b,
                                    sigma_m,
                                    panel,
                                    scan,
                                    beam);

    ASSERT_EQ(computed_bboxes.size(), num_reflections)
      << "Computed bboxes size mismatch";

    // Convert BoundingBoxExtents to flat array format
    std::vector<double> computed_bbox_data(num_reflections * 6);
    for (size_t i = 0; i < num_reflections; ++i) {
        computed_bbox_data[i * 6 + 0] = computed_bboxes[i].x_min;
        computed_bbox_data[i * 6 + 1] = computed_bboxes[i].x_max;
        computed_bbox_data[i * 6 + 2] = computed_bboxes[i].y_min;
        computed_bbox_data[i * 6 + 3] = computed_bboxes[i].y_max;
        computed_bbox_data[i * 6 + 4] = static_cast<double>(computed_bboxes[i].z_min);
        computed_bbox_data[i * 6 + 5] = static_cast<double>(computed_bboxes[i].z_max);
    }

    // Validate against existing bounding boxes from integrated reflections
    int result = validate_bounding_boxes(integrated_reflections,
                                         computed_bbox_data,
                                         num_reflections,
                                         10,
                                         integrated_data_path);
    EXPECT_EQ(result, 0) << "Validation failed";

    // Check for verbose mode via environment variable
    bool verbose = std::getenv("VERBOSE_TEST") != nullptr;

    // Track mismatches for summary
    size_t total_mismatches = 0;
    size_t x_min_mismatches = 0, x_max_mismatches = 0;
    size_t y_min_mismatches = 0, y_max_mismatches = 0;
    size_t z_min_mismatches = 0, z_max_mismatches = 0;

    // Compare computed bounding boxes with existing ones - fail if any differences
    for (size_t i = 0; i < num_reflections; ++i) {
        const auto &bbox = computed_bboxes[i];

        // Check that bounding boxes are reasonable
        EXPECT_LE(bbox.x_min, bbox.x_max) << "Invalid x bounds for reflection " << i;
        EXPECT_LE(bbox.y_min, bbox.y_max) << "Invalid y bounds for reflection " << i;
        EXPECT_LE(bbox.z_min, bbox.z_max) << "Invalid z bounds for reflection " << i;

        // Check that bounding boxes have non-zero size
        EXPECT_GT(bbox.x_max - bbox.x_min, 0) << "Zero width x for reflection " << i;
        EXPECT_GT(bbox.y_max - bbox.y_min, 0) << "Zero width y for reflection " << i;
        EXPECT_GT(bbox.z_max - bbox.z_min, 0) << "Zero width z for reflection " << i;

        // Compare with existing bbox - fail on any difference
        double ex_x_min = bbox_column(i, 0);
        double ex_x_max = bbox_column(i, 1);
        double ex_y_min = bbox_column(i, 2);
        double ex_y_max = bbox_column(i, 3);
        int ex_z_min = static_cast<int>(bbox_column(i, 4));
        int ex_z_max = static_cast<int>(bbox_column(i, 5));

        bool has_mismatch = false;

        if (bbox.x_min != ex_x_min) {
            x_min_mismatches++;
            has_mismatch = true;
            if (verbose) {
                EXPECT_EQ(bbox.x_min, ex_x_min)
                  << "x_min mismatch for reflection " << i;
            }
        }
        if (bbox.x_max != ex_x_max) {
            x_max_mismatches++;
            has_mismatch = true;
            if (verbose) {
                EXPECT_EQ(bbox.x_max, ex_x_max)
                  << "x_max mismatch for reflection " << i;
            }
        }
        if (bbox.y_min != ex_y_min) {
            y_min_mismatches++;
            has_mismatch = true;
            if (verbose) {
                EXPECT_EQ(bbox.y_min, ex_y_min)
                  << "y_min mismatch for reflection " << i;
            }
        }
        if (bbox.y_max != ex_y_max) {
            y_max_mismatches++;
            has_mismatch = true;
            if (verbose) {
                EXPECT_EQ(bbox.y_max, ex_y_max)
                  << "y_max mismatch for reflection " << i;
            }
        }
        if (bbox.z_min != ex_z_min) {
            z_min_mismatches++;
            has_mismatch = true;
            if (verbose) {
                EXPECT_EQ(bbox.z_min, ex_z_min)
                  << "z_min mismatch for reflection " << i;
            }
        }
        if (bbox.z_max != ex_z_max) {
            z_max_mismatches++;
            has_mismatch = true;
            if (verbose) {
                EXPECT_EQ(bbox.z_max, ex_z_max)
                  << "z_max mismatch for reflection " << i;
            }
        }

        if (has_mismatch) {
            total_mismatches++;
        }
    }

    // Print summary
    logger.info("\n=== Bounding Box Comparison Summary ===");
    logger.info("Total reflections checked: {}", num_reflections);
    logger.info("Reflections with mismatches: {}", total_mismatches);
    if (total_mismatches > 0) {
        logger.info("Mismatches by component:");
        logger.info("  x_min: {} mismatches", x_min_mismatches);
        logger.info("  x_max: {} mismatches", x_max_mismatches);
        logger.info("  y_min: {} mismatches", y_min_mismatches);
        logger.info("  y_max: {} mismatches", y_max_mismatches);
        logger.info("  z_min: {} mismatches", z_min_mismatches);
        logger.info("  z_max: {} mismatches", z_max_mismatches);
        logger.info("\nSet VERBOSE_TEST=1 to see detailed mismatch output");
    }

    // Fail test if any mismatches found
    EXPECT_EQ(total_mismatches, 0)
      << "Found " << total_mismatches << " reflections with bounding box mismatches";

    logger.info("Validation complete for {} reflections", num_reflections);
}

TEST(IntegratorTest, BasicTest) {
    EXPECT_TRUE(true);
}
