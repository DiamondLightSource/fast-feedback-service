/**
 * @file wait_utils.hpp
 * @brief Utilities for waiting on file/resource readiness.
 *
 * Provides functions for waiting on files or resources to become
 * ready for reading, with timeout support and visual feedback.
 */
#ifndef WAIT_UTILS_HPP
#define WAIT_UTILS_HPP

#include <fmt/core.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

/**
 * @brief Wait for a path to be ready for read, with visual feedback.
 *
 * Blocks until the checker function returns true for the given path,
 * or until the timeout is reached. Displays an animated progress
 * indicator while waiting.
 *
 * @param path The path to check for readiness.
 * @param checker Function that returns true when the path is ready.
 * @param timeout Maximum time to wait in seconds (default: 120.0).
 */
inline void wait_for_ready_for_read(const std::string &path,
                                    std::function<bool(const std::string &)> checker,
                                    float timeout = 120.0f) {
    if (!checker(path)) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto message_prefix =
          fmt::format("Waiting for \033[1;35m{}\033[0m to be ready for read", path);
        std::vector<std::string> ball = {
          "( ●    )",
          "(  ●   )",
          "(   ●  )",
          "(    ● )",
          "(     ●)",
          "(    ● )",
          "(   ●  )",
          "(  ●   )",
          "( ●    )",
          "(●     )",
        };
        int i = 0;
        while (!checker(path)) {
            auto wait_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                               std::chrono::high_resolution_clock::now() - start_time)
                               .count();
            fmt::print("\r{}  {} [{:4.1f} s] ", message_prefix, ball[i], wait_time);
            i = (i + 1) % ball.size();
            std::cout << std::flush;

            if (wait_time > timeout) {
                fmt::print("\nError: Waited too long for read availability\n");
                std::exit(1);
            }
            std::this_thread::sleep_for(80ms);
        }
        fmt::print("\n");
    }
}

#endif  // WAIT_UTILS_HPP
