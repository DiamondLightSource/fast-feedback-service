/**
 * @file integrator.cuh
 */

#pragma once

#include <cstdint>
#include <dx2/detector.hpp>

#include "math/device_precision.cuh"
#include "math/vector3d.cuh"

enum class FGAlgorithm : uint8_t { Ellipsoid, Dials };

struct DetectorParameters {
    scalar_t pixel_size[2];       // pixel pitch [fast, slow] in mm
    bool parallax_correction;     // whether to apply parallax correction
    scalar_t mu;                  // linear attenuation coefficient μ (mm⁻¹)
    scalar_t thickness;           // sensor thickness t₀ (mm)
    fastvec::Vector3D fast_axis;  // panel fast-axis direction f̂
    fastvec::Vector3D slow_axis;  // panel slow-axis direction ŝ
    fastvec::Vector3D origin;     // panel origin position (mm)
};

inline DetectorParameters make_detector_params(const Panel &panel) {
    DetectorParameters p;
    auto pixel_size = panel.get_pixel_size();
    p.pixel_size[0] = static_cast<scalar_t>(pixel_size[0]);
    p.pixel_size[1] = static_cast<scalar_t>(pixel_size[1]);
    p.parallax_correction = panel.has_parallax_correction();
    p.mu = static_cast<scalar_t>(panel.get_mu());
    p.thickness = static_cast<scalar_t>(panel.get_thickness());
    auto fast_axis = panel.get_fast_axis();
    auto slow_axis = panel.get_slow_axis();
    auto origin = panel.get_origin();
    p.fast_axis = fastvec::make_vector3d(static_cast<scalar_t>(fast_axis.x()),
                                         static_cast<scalar_t>(fast_axis.y()),
                                         static_cast<scalar_t>(fast_axis.z()));
    p.slow_axis = fastvec::make_vector3d(static_cast<scalar_t>(slow_axis.x()),
                                         static_cast<scalar_t>(slow_axis.y()),
                                         static_cast<scalar_t>(slow_axis.z()));
    p.origin = fastvec::make_vector3d(static_cast<scalar_t>(origin.x()),
                                      static_cast<scalar_t>(origin.y()),
                                      static_cast<scalar_t>(origin.z()));
    return p;
}
