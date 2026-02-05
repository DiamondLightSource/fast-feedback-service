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

#include "ffs_logger.hpp"
#include "kabsch.cuh"
#include "math/vector3d.cuh"

// Test fixture for Kabsch tests with file paths
class KabschTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Get paths relative to test directory
        test_dir = std::filesystem::path(__FILE__).parent_path();
        predicted_refl = test_dir / "data" / "predicted.refl";
        indexed_expt = test_dir / "data" / "indexed.expt";
    }

    std::filesystem::path test_dir;
    std::filesystem::path predicted_refl;
    std::filesystem::path indexed_expt;
};

TEST_F(KabschTest, ComputeKabschTransform) {
    // Check if test files exist
    ASSERT_TRUE(std::filesystem::exists(predicted_refl))
      << "Reflection file not found: " << predicted_refl;
    ASSERT_TRUE(std::filesystem::exists(indexed_expt))
      << "Experiment file not found: " << indexed_expt;

    // Load reflection data
    logger.info("Loading reflection data from file: {}", predicted_refl.string());
    ReflectionTable reflections(predicted_refl.string());

    const std::string data_path = "";

    // Extract required columns
    auto s1_vectors_opt = reflections.column<double>(data_path + "s1");
    ASSERT_TRUE(s1_vectors_opt.has_value())
      << "Column 's1' not found in reflection data.";
    auto s1_vectors = *s1_vectors_opt;

    auto phi_column_opt = reflections.column<double>(data_path + "xyzcal.mm");
    ASSERT_TRUE(phi_column_opt.has_value())
      << "Column 'xyzcal.mm' not found for phi positions.";
    auto phi_column = *phi_column_opt;

    size_t num_reflections = s1_vectors.extent(0);
    logger.info("Processing {} reflections for Kabsch transformation", num_reflections);

    // Parse experiment file
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

    Eigen::Vector3d s0_eigen = beam.get_s0();
    Eigen::Vector3d rotation_axis_eigen = gonio.get_rotation_axis();
    double wl = beam.get_wavelength();
    double osc_start = scan.get_oscillation()[0];
    double osc_width = scan.get_oscillation()[1];
    int image_range_start = scan.get_image_range()[0];

    // Convert to CUDA format
    fastvec::Vector3D s0_cuda =
      fastvec::make_vector3d(s0_eigen.x(), s0_eigen.y(), s0_eigen.z());
    fastvec::Vector3D rotation_axis_cuda = fastvec::make_vector3d(
      rotation_axis_eigen.x(), rotation_axis_eigen.y(), rotation_axis_eigen.z());

    logger.info("Testing Kabsch transformation for first reflection");

    // Test on the first reflection
    size_t refl_id = 0;

    // Get reflection centroid data
    Eigen::Vector3d s1_c_eigen(
      s1_vectors(refl_id, 0), s1_vectors(refl_id, 1), s1_vectors(refl_id, 2));
    fastvec::Vector3D s1_c_cuda =
      fastvec::make_vector3d(s1_c_eigen.x(), s1_c_eigen.y(), s1_c_eigen.z());
    double phi_c = phi_column(refl_id, 2);

    logger.info("Reflection {} centroid: s1=({:.6f}, {:.6f}, {:.6f}), phi={:.6f}",
                refl_id,
                s1_c_eigen.x(),
                s1_c_eigen.y(),
                s1_c_eigen.z(),
                phi_c);

    // Create test voxels around the reflection centroid
    std::vector<fastvec::Vector3D> test_s_pixels;
    std::vector<scalar_t> test_phi_pixels;

    // Test with a few voxel positions
    const int num_test_voxels = 5;
    for (int i = 0; i < num_test_voxels; ++i) {
        // Create slight variations around the centroid
        double x_offset = 100.0 + i * 10.0;  // pixels
        double y_offset = 200.0 + i * 10.0;  // pixels
        int z_offset = i;                    // frames

        // Convert pixel coordinates to lab frame
        std::array<double, 2> xy_mm = panel.px_to_mm(x_offset, y_offset);
        Eigen::Vector3d lab_coord =
          panel.get_d_matrix() * Eigen::Vector3d(xy_mm[0], xy_mm[1], 1.0);

        // Convert to reciprocal space vector
        Eigen::Vector3d s_pixel_eigen = lab_coord.normalized() / wl;
        fastvec::Vector3D s_pixel_cuda = fastvec::make_vector3d(
          s_pixel_eigen.x(), s_pixel_eigen.y(), s_pixel_eigen.z());

        // Calculate phi for this voxel
        double phi_pixel =
          osc_start + (z_offset - image_range_start + 1.5) * osc_width / 180.0 * M_PI;

        test_s_pixels.push_back(s_pixel_cuda);
        test_phi_pixels.push_back(static_cast<scalar_t>(phi_pixel));

        logger.info("Test voxel {}: pixel=({:.1f}, {:.1f}, {}), phi={:.6f}",
                    i,
                    x_offset,
                    y_offset,
                    z_offset,
                    phi_pixel);
    }

    // Allocate output arrays
    std::vector<fastvec::Vector3D> kabsch_results(num_test_voxels);
    std::vector<scalar_t> s1_len_results(num_test_voxels);

    // Call CUDA Kabsch transformation
    logger.info("Calling compute_kabsch_transform with {} test voxels",
                num_test_voxels);
    compute_kabsch_transform(test_s_pixels.data(),
                             test_phi_pixels.data(),
                             s1_c_cuda,
                             static_cast<scalar_t>(phi_c),
                             s0_cuda,
                             rotation_axis_cuda,
                             kabsch_results.data(),
                             s1_len_results.data(),
                             num_test_voxels);

    logger.info("Kabsch transformation completed successfully");

    // Validate results
    for (int i = 0; i < num_test_voxels; ++i) {
        const auto &eps = kabsch_results[i];
        scalar_t s1_len = s1_len_results[i];

        logger.info("Voxel {}: Kabsch coords ε=({:.6f}, {:.6f}, {:.6f}), |s₁|={:.6f}",
                    i,
                    eps.x,
                    eps.y,
                    eps.z,
                    s1_len);

        // Check results

        // Check that s1 length is positive
        EXPECT_GT(s1_len, 0.0f) << "|s₁| should be positive for voxel " << i;
    }

    logger.info("Kabsch transformation test completed successfully");
}
