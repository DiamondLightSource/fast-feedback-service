/**
 * @file thread_count.cc
 * @brief Implementation of the shared worker-thread count heuristic.
 *
 * sched_getaffinity is the only source that reflects a cpuset the
 * process has been confined to; hardware_concurrency() reports the
 * whole machine regardless, which over-subscribes inside a container.
 * The affinity call is Linux-only, so the fallback carries non-Linux
 * builds.
 */
#include "thread_count.hpp"

#include <thread>

#include "ffs_logger.hpp"

#ifdef __linux__
#include <sched.h>
#endif

uint32_t auto_select_thread_count() {
#ifdef __linux__
    cpu_set_t cpus;
    if (sched_getaffinity(0, sizeof(cpus), &cpus) == 0) {
        int count = CPU_COUNT(&cpus);
        logger.debug("sched_getaffinity reports {} allowed CPU(s)", count);
        if (count > 0) {
            return static_cast<uint32_t>(count);
        }
    } else {
        logger.debug("sched_getaffinity failed, falling back to hardware_concurrency");
    }
#endif
    uint32_t hw_threads = std::thread::hardware_concurrency();
    logger.debug("std::thread::hardware_concurrency() reports {} thread(s)",
                 hw_threads);
    return hw_threads ? hw_threads : 1;
}
