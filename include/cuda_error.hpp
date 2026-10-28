/**
 * @file cuda_error.hpp
 * @brief One error-reporting convention for every CUDA runtime call.
 *
 * The runtime reports a failure twice: as the return value of the call that
 * failed, and in a per-thread latch holding the most recent error, which
 * cudaGetLastError() reads and clears. The return value is the one to check,
 * since it belongs to a single call and cannot be stale. Nothing clears the
 * latch on success, so an unconsumed failure survives to be blamed on a later
 * call.
 *
 * A kernel launch is the exception, having no return value for <<<>>> to hand
 * back, so the latch is the only channel it has. Launch-time failures are all
 * it carries: a fault inside a running kernel surfaces later, as the return of
 * the next synchronising call, which is why a launch is checked twice.
 *
 * A device-side fault also destroys the context, after which every call in the
 * process fails identically until it exits. cuda_error carries the code that
 * identifies that case but does not yet classify it.
 */

#pragma once

#include <cuda_runtime.h>
#include <fmt/core.h>

#include <source_location>
#include <stdexcept>
#include <string>

/**
 * @brief An exception carrying the CUDA runtime error code that caused it.
 */
class cuda_error : public std::runtime_error {
  public:
    /**
     * @param code The failing cudaError_t.
     * @param message Human-readable description, including the call site.
     */
    cuda_error(cudaError_t code, const std::string &message)
        : std::runtime_error(message), _code(code) {}

    /// @brief The runtime error code behind the throw.
    auto code() const noexcept -> cudaError_t {
        return _code;
    }

  private:
    cudaError_t _code;
};

/// @brief Format an error code as "cudaErrorName: description".
inline auto cuda_error_string(cudaError_t err) -> std::string {
    return fmt::format("{}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
}

/**
 * @brief Throw if a CUDA runtime call returned a failure.
 *
 * Wraps the call itself, so the error is attributed to it and the latch is
 * left clean for the next launch check.
 *
 * @param err The return value of the call being checked.
 * @param location Defaulted; resolves to the call site, not to this function.
 * @throws cuda_error If err is anything other than cudaSuccess.
 */
inline void cuda_check(
  cudaError_t err,
  const std::source_location location = std::source_location::current()) {
    if (err != cudaSuccess) {
        throw cuda_error(err,
                         fmt::format("{}:{}: {}",
                                     location.file_name(),
                                     location.line(),
                                     cuda_error_string(err)));
    }
}

/**
 * @brief Throw if the immediately preceding kernel launch failed to start.
 *
 * Reads and clears the last-error latch, which is the only channel a launch
 * has. Catches an invalid launch configuration; a fault inside the kernel
 * arrives later, at the next synchronising call.
 *
 * @param kernel_name The launched kernel, for the message.
 * @param location Defaulted; resolves to the call site, not to this function.
 * @throws cuda_error If the launch was rejected.
 */
inline void cuda_check_launch(
  const char *kernel_name,
  const std::source_location location = std::source_location::current()) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw cuda_error(err,
                         fmt::format("{}:{}: launch of {} failed: {}",
                                     location.file_name(),
                                     location.line(),
                                     kernel_name,
                                     cuda_error_string(err)));
    }
}
