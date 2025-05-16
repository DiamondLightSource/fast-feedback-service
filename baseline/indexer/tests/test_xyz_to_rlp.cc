#include <gtest/gtest.h>
#include <math.h>

#include <Eigen/Dense>
#include <dx2/beam.hpp>
#include <dx2/detector.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <dx2/scan.hpp>
#include <nlohmann/json.hpp>

#include "xyz_to_rlp.cc"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using json = nlohmann::json;

TEST(BaselineIndexer, XyztoRlptest) {
    // Test the xyz_to_rlp function. Compare output to the dials
    // equivalent on the same data: centroid_px_to_mm plus
    // map_centroids_to_reciprocal_space
    std::vector<double> xyzobs_px_data{{10.1, 10.1, 50.2, 20.1, 20.1, 70.2}};
    mdspan_type<double> xyzobs_px =
      mdspan_type<double>(xyzobs_px_data.data(), xyzobs_px_data.size() / 3, 3);
    Scan scan{{1, 100}, {0.0, 0.1}};  //image range and oscillation.
    Goniometer gonio{
      {{1.0, 0.0, 0.0}}, {{0.0}}, {{"phi"}}, 0};  //axes, angles, names, scan-axis.
    json panel_data;
    panel_data["fast_axis"] = {1.0, 0.0, 0.0};
    panel_data["slow_axis"] = {0.0, -1.0, 0.0};
    panel_data["origin"] = {-150, 162, -200};
    panel_data["pixel_size"] = {0.075, 0.075};
    panel_data["image_size"] = {4148, 4362};
    panel_data["trusted_range"] = {0.0, 46051};
    panel_data["type"] = std::string("SENSOR_PAD");
    panel_data["name"] = std::string("test");
    panel_data["raw_image_offset"] = {0, 0};
    panel_data["thickness"] = 0.45;
    panel_data["material"] = "Si";
    panel_data["mu"] = 3.92;
    panel_data["gain"] = 1.0;
    panel_data["pedestal"] = 0.0;
    json pxdata;
    pxdata["type"] = std::string("ParallaxCorrectedPxMmStrategy");
    panel_data["px_mm_strategy"] = pxdata;
    Panel panel{panel_data};      // use defaults
    MonochromaticBeam beam{1.0};  //wavelength

    xyz_to_rlp_results results = xyz_to_rlp(xyzobs_px, panel, beam, scan, gonio);
    // Check against the equivalent results from the dials calculation
    Vector3d expected_0{{-0.5021752936083477, 0.5690514955867707, 0.27788051106787137}};
    Vector3d expected_1{{-0.5009709068399325, 0.5770958485799975, 0.2562207980973077}};
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(results.rlp(0, i), expected_0[i]);
        EXPECT_DOUBLE_EQ(results.rlp(1, i), expected_1[i]);
    }
}