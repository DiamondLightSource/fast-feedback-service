#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "h5read.h"
#include "integrator.cuh"
#include "integrator/background.hpp"
#include "integrator/background_pool.hpp"
#include "integrator/extent.hpp"
#include "math/device_precision.cuh"
#include "math/vector3d.cuh"

using namespace fastvec;

struct Vec2D {
    scalar_t x;
    scalar_t y;
};

struct Mat2D {
    scalar_t xx, xy;
    scalar_t yx, yy;
};


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
                        const Vec2D *d_mubar,
                        const Mat2D *d_sbar_inv,
                        size_t num_reflections_this_image,
                        const uint8_t *d_mask,
                        uint32_t max_corner_tile_corners,
                        accumulator_t *d_foreground_sum,
                        uint32_t *d_foreground_count,
                        BackgroundPool background_pool,
                        uint32_t *d_background_spill,
                        unsigned long long *d_intensity_times_x,
                        unsigned long long *d_intensity_times_y,
                        uint8_t *d_success,
                    cudaStream_t stream);