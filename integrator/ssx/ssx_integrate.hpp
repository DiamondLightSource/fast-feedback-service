#pragma once
#include <Eigen/Dense>
#include <dx2/detector.hpp>
#include <vector>
#include <experimental/mdspan>
#include "mosaicity_parameterisation.hpp"
#include "predictor/predict.hpp"

using Vector3d = Eigen::Vector3d;
using Vector2d = Eigen::Vector2d;

template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;



Simple6MosaicityParameterisation refine_mosaicity(
  mdspan_type<const double> &xyzcal_px,
  mdspan_type<const double> &xyzobs_px,
  mdspan_type<const double> &covariances,
  mdspan_type<const int> &intensities,
  mdspan_type<const int> &miller_indices,
  mdspan_type<const double> &mobs,
  const Eigen::Vector3d &s0,
  const Panel &panel,
  const Eigen::Matrix3d &A);

std::vector<Prediction> predict_ssx(const Simple6MosaicityParameterisation &model,
                                    const Vector3d &s0,
                                    const Panel &panel,
                                    const Matrix3d &A);
/*void ssx_integrate(mdspan_type<const double> &xyzcal_px,
                   mdspan_type<const double> &xyzobs_px,
                   mdspan_type<const double> &covariances,
                   mdspan_type<const double> &intensities,
                   mdspan_type<const int> &miller_indices,
                   mdspan_type<const double> &mobs,
                   const Vector3d &s0,
                   const Panel &panel,
                   const Matrix3d &A);*/