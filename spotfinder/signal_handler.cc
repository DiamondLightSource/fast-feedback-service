/**
 * @file signal_handler.cc
 * @brief Implementation of global signal handling for spotfinder application.
 */
#include "signal_handler.hpp"

#include <fmt/core.h>

#include <cstdlib>

// Global stop token for picking up user cancellation
std::stop_source global_stop;

// Function for passing to std::signal to register the stop request
extern "C" void stop_processing(int sig) {
    if (global_stop.stop_requested()) {
        // We already requested before, but we want it faster. Abort.
        std::quick_exit(1);
    } else {
        fmt::print("Running interrupted by user request\n");
        global_stop.request_stop();
    }
}
