/**
 * @file cuda_arg_parser.cc
 * @brief Implementation of CUDA-specific argument parser for
 * GPU-accelerated applications.
 *
 * This file implements the CUDAArgumentParser class, which extends the
 * base argument parser to handle CUDA device selection and management.
 * It provides automatic device initialization, validation, and property
 * retrieval as part of the command-line argument parsing process.
 *
 * Key CUDA-specific features include:
 * - Device selection via command-line arguments
 * - Device listing utility for discovering available GPUs
 * - Automatic device initialization and validation
 * - Device property retrieval and display
 * - Early error detection for CUDA-related issues
 *
 * The parser ensures that the selected CUDA device is properly
 * configured before the main application logic begins execution.
 */
#include "cuda_arg_parser.hpp"

#include <cuda_runtime.h>
#include <fmt/core.h>

#include <cstdlib>

CUDAArgumentParser::CUDAArgumentParser(std::string version)
    : FFSArgumentParser(version) {
    add_argument("-d", "--device")
      .help("CUDA device index")
      .default_value(0)
      .action([&](const std::string &val) {
          _cuda_args.device_index = std::stoi(val);
          return _cuda_args.device_index;
      });

    add_argument("--list-devices")
      .help("List CUDA devices and exit")
      .implicit_value(false)
      .action([](const std::string &) {
          // Query available CUDA devices
          int count;
          cudaGetDeviceCount(&count);

          // Display device information and exit
          for (int i = 0; i < count; ++i) {
              cudaDeviceProp prop;
              cudaGetDeviceProperties(&prop, i);
              fmt::print("{}: {} (CUDA {}.{})\n", i, prop.name, prop.major, prop.minor);
          }
          std::exit(0);
      });
}

void CUDAArgumentParser::post_parse() {
    // Attempt to select the specified CUDA device
    if (cudaSetDevice(_cuda_args.device_index) != cudaSuccess) {
        fmt::print("\033[1;31mError: Could not select CUDA device\033[0m\n");
        std::exit(1);
    }

    // Retrieve device properties for validation and reporting
    if (cudaGetDeviceProperties(&_cuda_args.device, _cuda_args.device_index)
        != cudaSuccess) {
        fmt::print("\033[1;31mError: Could not get device properties\033[0m\n");
        std::exit(1);
    }

    // Display selected device information
    fmt::print("Using {} (CUDA {}.{})\n",
               _cuda_args.device.name,
               _cuda_args.device.major,
               _cuda_args.device.minor);
}

auto CUDAArgumentParser::parse_args(int argc, char **argv) -> CUDAArguments {
    // Parse base arguments first
    FFSArgumentParser::parse_args(argc, argv);

    // Copy common arguments to CUDA-specific structure
    _cuda_args.verbose = _arguments.verbose;

    return _cuda_args;
}