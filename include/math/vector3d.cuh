/**
 * @file vector3d.cuh
 * @brief CUDA-compatible 3D vector mathematics library
 * 
 * This header defines a lightweight Vector3D class for both CPU and GPU execution
 * contexts. The class provides essential 3D vector operations required for crystallographic
 * data processing.
 * 
 * Mathematical Operations:
 * - Vector arithmetic: +, -, * (scalar)
 * - Geometric operations: dot(), cross(), norm(), normalized()
 * - Data access: span() for mdspan integration
 * 
 * Usage Examples:
 * ```cpp
 * fastfb::Vector3D v1(1.0, 2.0, 3.0);
 * fastfb::Vector3D v2(4.0, 5.0, 6.0);
 * 
 * // Basic operations
 * fastfb::Vector3D sum = v1 + v2;
 * double dot_product = v1.dot(v2);
 * fastfb::Vector3D cross_product = v1.cross(v2);
 * fastfb::Vector3D unit_vector = v1.normalized();
 * 
 * // Access components via mdspan
 * auto span = v1.span();
 * double x_component = span[0];
 * ```
 * 
 * @author Dimitrios Vlachos
 * @date 2025
 * @see kabsch.cu for usage in coordinate transformations
 * @see integrator.cc for application implementation
 */

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <experimental/mdspan>

namespace fastfb {
/**
     * @brief 3D vector structure for mathematical operations
     * 
     * Provides basic 3D vector operations that can be used both on host
     * and device (GPU). All operations are marked with __host__ __device__
     * to enable compilation for both execution contexts.
     */
struct Vector3D {
    double x, y, z;  ///< X, Y, Z components of the vector

    /**
         * @brief Default constructor initializes vector to zero
         * Creates vector (0, 0, 0)
         */
    __host__ __device__ Vector3D() : x(0.0), y(0.0), z(0.0) {}

    /**
         * @brief Construct vector with specified components
         * Creates vector (x_, y_, z_)
         * @param x_ X component
         * @param y_ Y component  
         * @param z_ Z component
         */
    __host__ __device__ Vector3D(double x_, double y_, double z_)
        : x(x_), y(y_), z(z_) {}

    /**
         * @brief Get a 1D mdspan view of the vector components
         * Provides array-like access: v.span()[0] = x, v.span()[1] = y, v.span()[2] = z
         * @return mdspan view with compile-time extent of 3
         */
    __host__ __device__ auto span()
      -> std::experimental::mdspan<double, std::experimental::extents<size_t, 3>> {
        return std::experimental::mdspan<double, std::experimental::extents<size_t, 3>>(
          &x);
    }

    /**
         * @brief Get a const 1D mdspan view of the vector components  
         * @return const mdspan view with compile-time extent of 3
         */
    __host__ __device__ auto span() const
      -> std::experimental::mdspan<const double,
                                   std::experimental::extents<size_t, 3>> {
        return std::experimental::mdspan<const double,
                                         std::experimental::extents<size_t, 3>>(&x);
    }

    /**
         * @brief Vector addition operator
         * Computes: **a** + **b** = (a.x + b.x, a.y + b.y, a.z + b.z)
         * @param b Vector to add
         * @return Sum of this vector and b
         */
    __host__ __device__ Vector3D operator+(const Vector3D& b) const {
        return Vector3D(x + b.x, y + b.y, z + b.z);
    }

    /**
         * @brief Vector subtraction operator
         * Computes: **a** - **b** = (a.x - b.x, a.y - b.y, a.z - b.z)
         * @param b Vector to subtract
         * @return Difference of this vector and b
         */
    __host__ __device__ Vector3D operator-(const Vector3D& b) const {
        return Vector3D(x - b.x, y - b.y, z - b.z);
    }

    /**
         * @brief Scalar multiplication operator
         * Computes: s × **v** = (s·v.x, s·v.y, s·v.z)
         * @param scalar Scalar value to multiply by
         * @return Vector scaled by the scalar value
         */
    __host__ __device__ Vector3D operator*(double scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }

    /**
         * @brief Compute dot product with another vector
         * Computes: **a** · **b** = a.x·b.x + a.y·b.y + a.z·b.z
         * @param b Vector to compute dot product with
         * @return Dot product (scalar value)
         */
    __host__ __device__ double dot(const Vector3D& b) const {
        return x * b.x + y * b.y + z * b.z;
    }

    /**
         * @brief Compute cross product with another vector
         * Computes:
         * **a** × **b** = (a.y·b.z - a.z·b.y, a.z·b.x - a.x·b.z, a.x·b.y - a.y·b.x)
         * @param b Vector to compute cross product with
         * @return Cross product vector perpendicular to both inputs
         */
    __host__ __device__ Vector3D cross(const Vector3D& b) const {
        return Vector3D(y * b.z - z * b.y,  // i component: a.y·b.z - a.z·b.y
                        z * b.x - x * b.z,  // j component: a.z·b.x - a.x·b.z
                        x * b.y - y * b.x   // k component: a.x·b.y - a.y·b.x
        );
    }

    /**
         * @brief Compute the magnitude (length) of the vector
         * Computes: ||**v**|| = √(v.x² + v.y² + v.z²)
         * @return Euclidean norm of the vector
         */
    __host__ __device__ double norm() const {
        return sqrt(x * x + y * y + z * z);
    }

    /**
         * @brief Get a normalized (unit length) version of this vector
         * Computes:
         * **û** = **v** / ||**v**|| = (v.x/||**v**||, v.y/||**v**||, v.z/||**v**||)
         * @return Unit vector in the same direction, or zero vector if norm is zero
         */
    __host__ __device__ Vector3D normalized() const {
        double n = norm();
        // Handle zero-length vector case to avoid division by zero
        return n > 0.0 ? Vector3D(x / n, y / n, z / n) : Vector3D(0.0, 0.0, 0.0);
    }
};
}  // namespace fastfb