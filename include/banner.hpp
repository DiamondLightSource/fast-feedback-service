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
