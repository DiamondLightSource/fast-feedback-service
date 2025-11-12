#include <Eigen/Dense>
#include <nlohmann/json.hpp>
using Eigen::Matrix3d;
using Eigen::Vector3d;
using json = nlohmann::json;
#pragma once
/**
 * @brief A class to store the axis of rotation and return a rotation matrix for a given angle using the Rodriguez formula.
 * 
 */
class Rotator {
  private:
    Vector3d axis_ = Vector3d{0, 0, 0};

  public:
    Rotator(const Vector3d& axis) : axis_(axis.normalized()) {}

    /**
	 * @brief Output a rotation matrix corresponding to a given axis and angle.
	 *
	 * @param θ The angle of rotation (in degrees)
	 */
    Matrix3d rotation_matrix(double θ) const {
        // Convert to radians
        θ = θ * M_PI / 180;
        return Matrix3d{{cos(θ) + axis_(0) * axis_(0) * (1 - cos(θ)),
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
	 *
	 */
    Vector3d rotate(const Vector3d& vec, double θ) const {
        return rotation_matrix(θ) * vec;
    }

    /**
	 * @brief Multiply the rotation matrix (angle θ) by a 3x3 matrix and return the result.
	 *
	 */
    Matrix3d rotate(const Matrix3d& mat, double θ) const {
        return rotation_matrix(θ) * mat;
    }
};

/**
 * @brief Takes in a json list of numbers and returns a 3x3 matrix representation of it.
 * 
 * @param matrix_json The json object, a list of numbers with length 9.
 * @return Matrix3d 
 */
Matrix3d matrix_3d_from_json(json matrix_json) {
    return Matrix3d{{matrix_json[0], matrix_json[1], matrix_json[2]},
                    {matrix_json[3], matrix_json[4], matrix_json[5]},
                    {matrix_json[6], matrix_json[7], matrix_json[8]}};
}

/**
 * @brief Takes in a json list of numbers and returns a 3D vector representation of it.
 * 
 * @param vector_json The json object, a list of numbers with length 3.
 * @return Vector3d 
 */
Vector3d vector_3d_from_json(json vector_json) {
    return Vector3d{{vector_json[0], vector_json[1], vector_json[2]}};
}

/**
 * @brief A struct to store information about the predicted reflected ray.
 * 
 */
struct Ray {
    Vector3d s1;
    double angle;
    bool entering;
};
