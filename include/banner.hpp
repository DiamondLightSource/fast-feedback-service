/**
 * @file banner.hpp
 * @brief Startup banner: "FAST" drawn as a diffraction pattern, coloured by
 *        spot intensity.
 */
#pragma once

#include <string_view>

/**
 * @brief Print the startup banner and a line naming the program and its
 *        version to stdout.
 *
 * @param program Name shown under the art, e.g. "spotfinder".
 */
void print_banner(std::string_view program);

/**
 * @brief Print the banner in every colour map, on both backgrounds, each
 *        under a line naming its FFS_BANNER_PALETTE index.
 *
 * Prints whether or not stdout is a terminal; NO_COLOR and COLORTERM still
 * decide how much colour is used.
 *
 * @param program Name shown under the art, e.g. "spotfinder".
 */
void print_banner_preview(std::string_view program);
