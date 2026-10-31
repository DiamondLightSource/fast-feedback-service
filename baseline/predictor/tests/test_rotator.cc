#include <gtest/gtest.h>

#include <random>

#include "predictor/utils.hpp"

// Rotations about random axes match the rotation written out on the vector
TEST(Rotator, RandomAxes) {
    // Fixed seed so a failure reproduces on every run
    std::mt19937_64 generator(67);
    std::normal_distribution<double> normal(0, 1);
    std::uniform_real_distribution<double> angle_range(-360, 360);
    double worst_error = 0;
    for (int i = 0; i < 100000; ++i) {
        // Random unit axis, vector and angle in degrees
        Eigen::Vector3d axis =
          Eigen::Vector3d{normal(generator), normal(generator), normal(generator)}
            .normalized();
        Eigen::Vector3d vector{normal(generator), normal(generator), normal(generator)};
        double angle = angle_range(generator);

        // Rotate the vector about the axis directly, without building a matrix
        double radians = angle * M_PI / 180;
        Eigen::Vector3d expected = vector * cos(radians)
                                   + axis.cross(vector) * sin(radians)
                                   + axis * axis.dot(vector) * (1 - cos(radians));

        // Track the largest component error over all samples
        Eigen::Vector3d rotated = Rotator(axis).rotate(vector, angle);
        worst_error = std::max(worst_error, (rotated - expected).cwiseAbs().maxCoeff());
    }
    EXPECT_LT(worst_error, 1e-12) << worst_error;
}
