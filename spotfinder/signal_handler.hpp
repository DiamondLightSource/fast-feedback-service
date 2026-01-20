/**
 * @file signal_handler.hpp
 * @brief Global signal handling for spotfinder application.
 *
 * Provides signal handling facilities to gracefully handle user
 * interruption (Ctrl+C) during image processing.
 */
#ifndef SIGNAL_HANDLER_HPP
#define SIGNAL_HANDLER_HPP

#include <stop_token>

/// Global stop source for picking up user cancellation
extern std::stop_source global_stop;

/// Signal handler function for registering stop requests
extern "C" void stop_processing(int sig);

#endif  // SIGNAL_HANDLER_HPP
