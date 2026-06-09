/**
 * @file utils.cc
 * @brief Shared helpers for spot prediction (JSON conversion).
 */

#include "predictor/utils.hpp"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using json = nlohmann::json;

Matrix3d matrix_3d_from_json(json matrix_json) {
    return Matrix3d{{matrix_json[0], matrix_json[1], matrix_json[2]},
                    {matrix_json[3], matrix_json[4], matrix_json[5]},
                    {matrix_json[6], matrix_json[7], matrix_json[8]}};
}

Vector3d vector_3d_from_json(json vector_json) {
    return Vector3d{{vector_json[0], vector_json[1], vector_json[2]}};
}
