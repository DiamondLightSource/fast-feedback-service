/**
 * @file test_connected_components.cc
 * @brief Unit tests for Reflection3D peak selection (connected_components).
 *
 * Tests tie-breaking in peak_centroid_distance(): when several pixels
 * share the maximum intensity, the peak is the pixel nearest the
 * centroid rather than the first in z, y, x order (see
 * https://github.com/dials/dials/issues/3014). The selected peak is
 * read back through the returned peak-centroid distance.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <optional>

#include "connected_components.hpp"

namespace {

/// Build a single-frame signal (z = 0) at integer pixel (x, y).
Signal make_signal(uint32_t x, uint32_t y, pixel_t intensity) {
    return Signal{x, y, std::optional<int>(0), intensity, 0};
}

/// Euclidean distance from a pixel centre to the given centroid.
float distance_to_com(uint32_t x,
                      uint32_t y,
                      int z,
                      float com_x,
                      float com_y,
                      float com_z) {
    float dx = (static_cast<float>(x) + 0.5f) - com_x;
    float dy = (static_cast<float>(y) + 0.5f) - com_y;
    float dz = (static_cast<float>(z) + 0.5f) - com_z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace

// Single brightest pixel -> the peak is that pixel and the reported
// distance is its distance to the centroid. Covers the common untied
// path.
TEST(ReflectionPeakSelection, UniqueMaximum) {
    Reflection3D reflection;
    reflection.add_signal(make_signal(0, 0, 2));
    reflection.add_signal(make_signal(3, 0, 8));  // unique maximum

    auto [com_x, com_y, com_z] = reflection.center_of_mass();
    float expected = distance_to_com(3, 0, 0, com_x, com_y, com_z);

    EXPECT_FLOAT_EQ(reflection.peak_centroid_distance(), expected);
}

// Two pixels share the maximum value. The intensity mass (and so the
// centroid) sits in the high-x, high-y corner, so the tie breaks to the
// pixel nearest the centroid, not the (0, 0) pixel that wins the old z,
// y, x ordering. See https://github.com/dials/dials/issues/3014.
TEST(ReflectionPeakSelection, TieResolvedByCentroid) {
    Reflection3D reflection;

    // Blob of intensity in the high-x, high-y corner so the centroid sits there.
    for (uint32_t y = 3; y <= 5; ++y) {
        for (uint32_t x = 3; x <= 5; ++x) {
            reflection.add_signal(make_signal(x, y, 5));
        }
    }
    reflection.add_signal(make_signal(0, 0, 10));  // far, z/y/x-first maximum
    reflection.add_signal(make_signal(5, 5, 10));  // near the centroid

    auto [com_x, com_y, com_z] = reflection.center_of_mass();
    float near_distance = distance_to_com(5, 5, 0, com_x, com_y, com_z);
    float far_distance = distance_to_com(0, 0, 0, com_x, com_y, com_z);

    // Check the near pixel is the closer of the two.
    ASSERT_LT(near_distance, far_distance);

    // The near-centroid pixel is selected, not the (0, 0) corner.
    EXPECT_FLOAT_EQ(reflection.peak_centroid_distance(), near_distance);
}
