/**
 * @file thread_count.hpp
 * @brief Default worker-thread count selection shared across FFS
 * applications.
 */
#pragma once

#include <cstdint>

/**
 * @brief Determine a sensible default number of worker threads.
 *
 * Prefers the number of CPUs the process is actually allowed to run on
 * (respecting cgroup/scheduler affinity for when in containers).
 * Falls back to std::thread::hardware_concurrency(), and finally to 1.
 *
 * @return The selected thread count, always at least 1.
 */
uint32_t auto_select_thread_count();
