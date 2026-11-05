#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "h5read.h"
#include "integrator.cuh"
#include "integrator/background.hpp"
#include "integrator/extent.hpp"
#include "math/device_precision.cuh"
#include "math/vector3d.cuh"

using namespace fastvec;

constexpr uint32_t ELLIPSOID_THREADS = 128;

void compute_ellipsoid_integrate(pixel_t *d_image,
                        size_t image_pitch,
                        uint32_t width,
                        uint32_t height,
                        int image_num,
                        const scalar_t *d_matrix,
                        const scalar_t wavelength,
                        const DetectorParameters det_params,
                        const Vector3D s0,
                        const BoundingBoxExtents *d_bboxes,
                        const size_t *d_reflection_indices,
                        const Vector3D *d_s2_vectors,
                        const scalar_t *sigma_matices,
                        size_t num_reflections_this_image,
                        const uint8_t *d_mask,
                        uint32_t max_corner_tile_corners,
                        accumulator_t *d_foreground_sum,
                        uint32_t *d_foreground_count,
                        uint8_t *d_success,
                    cudaStream_t stream);