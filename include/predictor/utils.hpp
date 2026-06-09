/**
 * @file utils.hpp
 * @brief Shared helpers for spot prediction (rotation, JSON conversion, Ray).
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <nlohmann/json.hpp>

/**
 * @brief A class to store the axis of rotation and return a rotation matrix for
 * a given angle using the Rodriguez formula.
 */
class Rotator {
  private:
    Eigen::Vector3d axis_ = Eigen::Vector3d{0, 0, 0};

  public:
    Rotator(const Eigen::Vector3d &axis) : axis_(axis.normalized()) {}

    /**
     * @brief Output a rotation matrix corresponding to a given axis and angle.
     *
     * @param θ The angle of rotation (in degrees)
     */
    Eigen::Matrix3d rotation_matrix(double θ) const {
        // Convert to radians
        θ = θ * M_PI / 180;
        return Eigen::Matrix3d{
          {cos(θ) + axis_(0) * axis_(0) * (1 - cos(θ)),
           -axis_(2) * sin(θ) + axis_(0) * axis_(1) * (1 - cos(θ)),
           axis_(1) * sin(θ) + axis_(0) * axis_(1) * (1 - cos(θ))},
          {axis_(2) * sin(θ) + axis_(0) * axis_(1) * (1 - cos(θ)),
           cos(θ) + axis_(1) * axis_(1) * (1 - cos(θ)),
           -axis_(0) * sin(θ) + axis_(1) * axis_(2) * (1 - cos(θ))},
          {-axis_(1) * sin(θ) + axis_(0) * axis_(2) * (1 - cos(θ)),
           axis_(0) * sin(θ) + axis_(1) * axis_(2) * (1 - cos(θ)),
           cos(θ) + axis_(2) * axis_(2) * (1 - cos(θ))}};
    }

    /**
     * @brief Rotate a 3D vector by a given angle around the pre-specified axis
     */
    Eigen::Vector3d rotate(const Eigen::Vector3d &vec, double θ) const {
        return rotation_matrix(θ) * vec;
    }

    /**
     * @brief Multiply the rotation matrix (angle θ) by a 3x3 matrix and return
     * the result.
     */
    Eigen::Matrix3d rotate(const Eigen::Matrix3d &mat, double θ) const {
        return rotation_matrix(θ) * mat;
    }
};

/**
 * @brief Takes in a json list of numbers and returns a 3x3 matrix
 * representation of it.
 *
 * @param matrix_json The json object, a list of numbers with length 9.
 */
Eigen::Matrix3d matrix_3d_from_json(nlohmann::json matrix_json);

/**
 * @brief Takes in a json list of numbers and returns a 3D vector representation
 * of it.
 *
 * @param vector_json The json object, a list of numbers with length 3.
 */
Eigen::Vector3d vector_3d_from_json(nlohmann::json vector_json);

/**
 * @brief A struct to store information about the predicted reflected ray.
 */
struct Ray {
    Eigen::Vector3d s1;
    double angle;
    bool entering;
};
