/**
 * @file ffs_hd.hpp
 * @brief Marks a function as compilable for both the host and the device.
 *
 * A shared algorithm core is written once and compiled twice: into a CUDA
 * translation unit, where it has to carry `__host__ __device__`, and into a
 * plain C++ one for the CPU baseline and the host tests, where those keywords
 * do not exist. FFS_HD, for fast feedback service host device, is whichever of
 * the two the current compiler needs, so one definition serves both and the two
 * builds cannot drift apart.
 *
 * It expands to nothing on the host rather than to `inline`, so a function
 * defined in a header needs an `inline` of its own to remain one definition
 * across translation units. The convention in this tree is to write
 * `FFS_HD inline`, and to reserve a bare FFS_HD for `constexpr` functions,
 * which are already implicitly inline.
 */

#pragma once

#if defined(__CUDACC__)
#define FFS_HD __host__ __device__
#else
#define FFS_HD
#endif
