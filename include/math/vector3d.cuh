/**
 * @file vector3d.cuh
 * @brief CUDA-compatible 3D vector mathematics library
 * 
 * This header provides 3D vector operations using CUDA's built-in vector types
 * (double3/float3) for optimal GPU performance. The implementation wraps CUDA's
 * native types with convenient mathematical operations.
 * 
 *  * Performance Design:
 * - Uses CUDA's hardware-optimized double3/float3 types directly
 * - Free functions avoid virtual function call overhead
 * - Maintains guaranteed memory layout for vectorized GPU operations
 * - Zero-overhead abstraction - compiles to optimal GPU assembly
 * - Direct interoperability with CUDA libraries (cuBLAS, Thrust, etc.)
 * 
 * Mathematical Operations:
 * - Vector arithmetic: +, -, * (scalar)
 * - Geometric operations: dot(), cross(), norm(), normalized()
 * - Data access: span() for mdspan integration
 * 
 * Precision Control:
 * The library supports compile-time precision switching via USE_DOUBLE_PRECISION:
 * - Default: single precision (float3) for optimal GPU performance
 * - With -DUSE_DOUBLE_PRECISION: double precision (double3) for accuracy
 * 
 * Usage Examples:
 * ```cpp
 * fastvec::Vector3D v1 = fastvec::make_vector3d(1.0, 2.0, 3.0);
 * fastvec::Vector3D v2 = fastvec::make_vector3d(4.0, 5.0, 6.0);
 * 
 * // Basic operations
 * fastvec::Vector3D sum = v1 + v2;
 * fastvec::scalar_t dot_product = fastvec::dot(v1, v2);
 * fastvec::Vector3D cross_product = fastvec::cross(v1, v2);
 * fastvec::Vector3D unit_vector = fastvec::normalized(v1);
 * 
 * // Access components via mdspan
 * auto span = fastvec::span(v1);
 * fastvec::scalar_t x_component = span[0];
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

namespace fastvec {

// Type configuration - change this to switch between float and double precision
#ifdef USE_DOUBLE_PRECISION
using Vector3D = double3;
using scalar_t = double;
#define MAKE_VECTOR3D make_double3
#define CUDA_SQRT sqrt
#else
using Vector3D = float3;
using scalar_t = float;
#define MAKE_VECTOR3D make_float3
#define CUDA_SQRT sqrtf
#endif

// Convenience aliases for mdspan types
using vector3d_span =
  std::experimental::mdspan<scalar_t, std::experimental::extents<size_t, 3>>;
using const_vector3d_span =
  std::experimental::mdspan<const scalar_t, std::experimental::extents<size_t, 3>>;

// Convenience function to create Vector3D
/**
 * @brief Create a Vector3D from scalar components
 * @param x X component
 * @param y Y component
 * @param z Z component
 * @return Vector3D with specified components
 */
__host__ __device__ inline Vector3D make_vector3d(scalar_t x, scalar_t y, scalar_t z) {
    return MAKE_VECTOR3D(x, y, z);
}

// Vector arithmetic operators
/**
 * @brief Vector addition operator
 * Computes: **a** + **b** = (a.x + b.x, a.y + b.y, a.z + b.z)
 * @param a First vector
 * @param b Second vector
 * @return Sum of the two vectors
 */
__host__ __device__ inline Vector3D operator+(const Vector3D& a, const Vector3D& b) {
    return MAKE_VECTOR3D(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Vector subtraction operator
 * Computes: **a** - **b** = (a.x - b.x, a.y - b.y, a.z - b.z)
 * @param a First vector
 * @param b Second vector
 * @return Difference of the two vectors
 */
__host__ __device__ inline Vector3D operator-(const Vector3D& a, const Vector3D& b) {
    return MAKE_VECTOR3D(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * @brief Scalar multiplication operator
 * Computes: s × **v** = (s·v.x, s·v.y, s·v.z)
 * @param v Vector to scale
 * @param scalar Scalar value to multiply by
 * @return Vector scaled by the scalar value
 */
__host__ __device__ inline Vector3D operator*(const Vector3D& v, scalar_t scalar) {
    return MAKE_VECTOR3D(v.x * scalar, v.y * scalar, v.z * scalar);
}

/**
 * @brief Scalar multiplication operator (commutative)
 * Computes: s × **v** = (s·v.x, s·v.y, s·v.z)
 * @param scalar Scalar value to multiply by
 * @param v Vector to scale
 * @return Vector scaled by the scalar value
 */
__host__ __device__ inline Vector3D operator*(scalar_t scalar, const Vector3D& v) {
    return MAKE_VECTOR3D(v.x * scalar, v.y * scalar, v.z * scalar);
}

/**
 * @brief Scalar division operator
 * Computes: **v** / s = (v.x/s, v.y/s, v.z/s)
 * @param v Vector to divide
 * @param scalar Scalar value to divide by
 * @return Vector divided by the scalar value
 */
__host__ __device__ inline Vector3D operator/(const Vector3D& v, scalar_t scalar) {
    return MAKE_VECTOR3D(v.x / scalar, v.y / scalar, v.z / scalar);
}

// Vector operations
/**
 * @brief Compute dot product of two vectors
 * Computes: **a** · **b** = a.x·b.x + a.y·b.y + a.z·b.z
 * @param a First vector
 * @param b Second vector
 * @return Dot product (scalar value)
 */
__host__ __device__ inline scalar_t dot(const Vector3D& a, const Vector3D& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Compute cross product of two vectors
 * Computes:
 * **a** × **b** = (a.y·b.z - a.z·b.y, a.z·b.x - a.x·b.z, a.x·b.y - a.y·b.x)
 * @param a First vector
 * @param b Second vector
 * @return Cross product vector perpendicular to both inputs
 */
__host__ __device__ inline Vector3D cross(const Vector3D& a, const Vector3D& b) {
    return MAKE_VECTOR3D(a.y * b.z - a.z * b.y,   // i component
                         a.z * b.x - a.x * b.z,   // j component
                         a.x * b.y - a.y * b.x);  // k component
}

/**
 * @brief Compute the magnitude (length) of a vector
 * Computes: ||**v**|| = √(v.x² + v.y² + v.z²)
 * @param v Vector to compute magnitude of
 * @return Euclidean norm of the vector
 */
__host__ __device__ inline scalar_t norm(const Vector3D& v) {
    return CUDA_SQRT(v.x * v.x + v.y * v.y + v.z * v.z);
}

/**
 * @brief Get a normalized (unit length) version of a vector
 * Computes:
 * **û** = **v** / ||**v**|| = (v.x/||**v**||, v.y/||**v**||, v.z/||**v**||)
 * @param v Vector to normalize
 * @return Unit vector in the same direction, or zero vector if norm is zero
 */
__host__ __device__ inline Vector3D normalized(const Vector3D& v) {
    scalar_t n = norm(v);
    // Handle zero-length vector case to avoid division by zero
    return n > 0.0 ? MAKE_VECTOR3D(v.x / n, v.y / n, v.z / n)
                   : MAKE_VECTOR3D(0.0, 0.0, 0.0);
}

/**
 * @brief Get a 1D mdspan view of the vector components
 * Provides array-like access: span(v)[0] = x, span(v)[1] = y, span(v)[2] = z
 * @param v Vector to get span view of
 * @return mdspan view with compile-time extent of 3
 */
__host__ __device__ inline auto span(Vector3D& v) -> vector3d_span {
    return vector3d_span(&v.x);
}

/**
 * @brief Get a const 1D mdspan view of the vector components  
 * @param v Vector to get const span view of
 * @return const mdspan view with compile-time extent of 3
 */
__host__ __device__ inline auto span(const Vector3D& v) -> const_vector3d_span {
    return const_vector3d_span(&v.x);
}
}  // namespace fastvec