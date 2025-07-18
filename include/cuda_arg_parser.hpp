/**
 * @file cuda_arg_parser.hpp
 * @brief CUDA-specific argument parser extension for GPU-accelerated
 * FFS applications.
 *
 * This file extends the base argument parser to handle CUDA-specific
 * command-line arguments and device management. It provides automatic
 * CUDA device selection, validation, and initialization as part of the
 * argument parsing process.
 *
 * The CUDA parser adds device selection capabilities, device listing
 * utilities, and automatic device property retrieval. It ensures that
 * the selected CUDA device is properly initialized and available before
 * application execution begins, providing early error detection for
 * GPU-related issues.
 *
 * @note This parser requires CUDA runtime to be available and properly
 * configured.
 */
#pragma once

#include <cuda_runtime.h>

#include "arg_parser.hpp"

/**
 * @brief Extended argument structure for CUDA-specific applications.
 *
 * This structure extends FFSArguments to include CUDA-specific
 * parameters such as device selection and device properties. It
 * maintains compatibility with the base argument structure while adding
 * GPU-related configuration.
 */
struct CUDAArguments : public FFSArguments {
    int device_index = 0;     ///< Selected CUDA device index (default: 0)
    cudaDeviceProp device{};  ///< Properties of the selected CUDA device
};

/**
 * @brief CUDA-aware argument parser for GPU-accelerated FFS
 * applications.
 *
 * This class extends FFSArgumentParser to handle CUDA-specific
 * command-line arguments such as device selection and device listing.
 * It automatically initializes the selected CUDA device and validates
 * device availability during the parsing process.
 */
class CUDAArgumentParser : public FFSArgumentParser {
  public:
    explicit CUDAArgumentParser(std::string version = "0.1.0");

    /**
     * @brief Parses command-line arguments and returns CUDA-specific
     * argument data.
     *
     * Processes command-line arguments using the base parser
     * functionality, then populates and returns a CUDAArguments
     * structure containing both common and CUDA-specific parsed values.
     *
     * @param argc Number of command-line arguments
     * @param argv Array of command-line argument strings
     * @return CUDAArguments Structure containing all parsed argument
     * values
     */
    auto parse_args(int argc, char** argv) -> CUDAArguments;

  protected:
    /**
     * @brief Post-parsing hook for CUDA device initialization and
     * validation.
     *
     * This method is called after argument parsing to set up the
     * selected CUDA device and retrieve its properties. It validates
     * device availability and displays device information to the user.
     */
    void post_parse() override;

    CUDAArguments _cuda_args;  ///< Internal storage for CUDA-specific arguments
};