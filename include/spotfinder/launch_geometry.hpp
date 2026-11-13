/**
 * @file launch_geometry.hpp
 * @brief The thread block shapes the spotfinder's kernels launch with, and
 *        the shared memory arithmetic that follows from them.
 *
 * The thresholding kernels run one thread per pixel and stage their inputs in
 * shared memory before reading them. A thread at the edge of a block of
 * bx x by threads reads neighbours up to r pixels beyond it, so each staged
 * input is a tile of (bx + 2r) x (by + 2r) pixels: the block's own pixels
 * plus a halo r deep on every side. At 32x16 with r = 3 that is
 * 38 x 22 = 836 pixels.
 *
 * A kernel has a single dynamic shared memory allocation, so the tiles sit
 * end to end within it, one per input, each holding the same number of
 * pixels in its own element type. The allocation is that pixel count times
 * the summed element sizes: the 16-bit first pass stages an 8-bit mask and a
 * 16-bit image, so it needs 836 x (1 + 2) = 2508 bytes.
 */

#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @brief Thread block for the thresholding and erosion kernels.
 *
 * At one warp wide, each warp covers a single contiguous run of one row.
 */
constexpr uint32_t THRESHOLD_BLOCK_X = 32;
constexpr uint32_t THRESHOLD_BLOCK_Y = 16;

/// @brief Thread block for the one-off resolution mask kernel.
constexpr uint32_t RESOLUTION_MASK_BLOCK_X = 32;
constexpr uint32_t RESOLUTION_MASK_BLOCK_Y = 32;

/**
 * @brief The number of pixels a block's shared tile holds, halo included.
 *
 * @param block_x Width of the thread block, in threads.
 * @param block_y Height of the thread block, in threads.
 * @param radius Halo depth on each side, in pixels.
 * @return The pixel count of one partition of the shared tile buffer.
 */
constexpr size_t shared_tile_pixels(uint32_t block_x,
                                    uint32_t block_y,
                                    uint32_t radius) {
    return static_cast<size_t>(block_x + 2 * radius) * (block_y + 2 * radius);
}

/**
 * @brief The bytes a kernel's shared tile buffer needs, one partition per type.
 *
 * @tparam Types One type per partition the kernel carves out of the buffer.
 * @param block_x Width of the thread block, in threads.
 * @param block_y Height of the thread block, in threads.
 * @param radius Halo depth on each side, in pixels.
 * @return The size of the shared memory request, in bytes.
 */
template <typename... Types>
constexpr size_t shared_tile_buffer_bytes(uint32_t block_x,
                                          uint32_t block_y,
                                          uint32_t radius) {
    return shared_tile_pixels(block_x, block_y, radius) * (sizeof(Types) + ...);
}
