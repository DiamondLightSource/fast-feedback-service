#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <fmt/core.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <type_traits>

#include "../../../common/include/common.hpp"

#if __has_include(<hdf5.h>)
#define HAS_HDF5
namespace _hdf5 {
#include <hdf5.h>
}
#endif

class cuda_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

auto cuda_error_string(cudaError_t err) {
    const char *err_name = cudaGetErrorName(err);
    const char *err_str = cudaGetErrorString(err);
    return fmt::format("{}: {}", std::string{err_name}, std::string{err_str});
}

/// Raise an exception IF CUDA is in an error state, with the name and description
auto cuda_throw_error() -> void {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw cuda_error(cuda_error_string(err));
    }
}

class CUDAArgumentParser;

struct CUDAArguments {
  public:
    bool verbose = false;
    std::string file;

    int device_index = 0;

    cudaDeviceProp device;

  private:
    friend class CUDAArgumentParser;
};

class CUDAArgumentParser : public argparse::ArgumentParser {
  public:
    CUDAArgumentParser(std::string version = "0.1.0")
        : ArgumentParser("", version, argparse::default_arguments::help) {
        this->add_argument("-v", "--verbose")
          .help("Verbose output")
          .implicit_value(false)
          .action([&](const std::string &value) { _arguments.verbose = true; });

        this->add_argument("-d", "--device")
          .help("Index of the CUDA device device to target.")
          .default_value(0)
          .metavar("INDEX")
          .action([&](const std::string &value) {
              _arguments.device_index = std::stoi(value);
              return _arguments.device_index;
          });
        this->add_argument("--list-devices")
          .help("List the order of CUDA devices, then quit.")
          .implicit_value(false)
          .action([](const std::string &value) {
              int deviceCount;
              if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
                  fmt::print("\033[1;31mError: Could not get GPU count ({})\033[0m\n",
                             cudaGetErrorString(cudaGetLastError()));
                  std::exit(1);
              }

              fmt::print("System GPUs:\n");
              for (int device = 0; device < deviceCount; ++device) {
                  cudaDeviceProp deviceProp;
                  cudaGetDeviceProperties(&deviceProp, device);
                  fmt::print("  {:2d}: {} (PCI {}:{}:{}, CUDA {}.{})\n",
                             device,
                             bold("{}", deviceProp.name),
                             deviceProp.pciDomainID,
                             deviceProp.pciBusID,
                             deviceProp.pciDeviceID,
                             deviceProp.major,
                             deviceProp.minor);
              }

              std::exit(0);
          });
    }

    auto parse_args(int argc, char **argv) -> CUDAArguments {
        // Convert these to std::string
        std::vector<std::string> args{argv, argv + argc};
        // Look for a "common.args" file in the current folder. If
        // present, add each line as an argument.
        std::ifstream common("common.args");
        std::filesystem::path argfile{"common.args"};
        if (std::filesystem::exists(argfile)) {
            fmt::print("File {} exists, loading default args:\n",
                       bold(argfile.string()));
            std::fstream f{argfile};
            std::string arg;
            while (std::getline(f, arg)) {
                if (arg.size() > 0) {
                    fmt::print("    {}\n", arg);
                    args.push_back(arg);
                }
            }
        }

        try {
            ArgumentParser::parse_args(args);
        } catch (std::runtime_error &e) {
            fmt::print("{}: {}\n{}\n",
                       bold(red("Error")),
                       red(e.what()),
                       ArgumentParser::usage());
            std::exit(1);
        }

        // cudaDeviceProp deviceProp;
        if (cudaSetDevice(_arguments.device_index) != cudaSuccess) {
            fmt::print(red("{}: Could not select device ({})"),
                       bold("Error"),
                       cuda_error_string(cudaGetLastError()));
            std::exit(1);
        }
        if (cudaGetDeviceProperties(&_arguments.device, _arguments.device_index)
            != cudaSuccess) {
            fmt::print(red("{}: Could not inspect GPU ({})\n",
                           bold("Error"),
                           cuda_error_string(cudaGetLastError())));
            std::exit(1);
        }
        fmt::print("Using {} (CUDA {}.{})\n\n",
                   bold(_arguments.device.name),
                   _arguments.device.major,
                   _arguments.device.minor);

#ifdef HAS_HDF5
        // If we activated h5read, then handle hdf5 verbosity
        if (_activated_h5read && !_arguments.verbose) {
            _hdf5::H5Eset_auto((_hdf5::hid_t)0, NULL, NULL);
        }
#endif

        return _arguments;
    }

    void add_h5read_arguments() {
        bool implicit_sample = std::getenv("H5READ_IMPLICIT_SAMPLE") != NULL;

        auto &group = add_mutually_exclusive_group(!implicit_sample);
        group.add_argument("--sample")
          .help(
            "Don't load a data file, instead use generated test data. If "
            "H5READ_IMPLICIT_SAMPLE is set, then this is assumed, if a file is not "
            "provided.")
          .implicit_value(true);
        group.add_argument("file")
          .metavar("FILE.nxs")
          .help("Path to the Nexus file to parse")
          .action([&](const std::string &value) { _arguments.file = value; });
        _activated_h5read = true;
    }

  private:
    CUDAArguments _arguments{};
    bool _activated_h5read = false;
};

template <typename T>
auto make_cuda_malloc(size_t num_items = 1) {
    using Tb = typename std::remove_extent<T>::type;
    Tb *obj = nullptr;
    if (cudaMalloc(&obj, sizeof(Tb) * num_items) != cudaSuccess || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](Tb *ptr) { cudaFree(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}

template <typename T>
auto make_cuda_managed_malloc(size_t num_items) {
    using Tb = typename std::remove_extent<T>::type;
    Tb *obj = nullptr;
    if (cudaMallocManaged(&obj, sizeof(Tb) * num_items) != cudaSuccess
        || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](Tb *ptr) { cudaFree(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}

/// Allocate memory using cudaMallocHost
template <typename T>
auto make_cuda_pinned_malloc(size_t num_items = 1) {
    using Tb = typename std::remove_extent<T>::type;
    Tb *obj = nullptr;
    if (cudaMallocHost(&obj, sizeof(Tb) * num_items) != cudaSuccess || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](Tb *ptr) { cudaFreeHost(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}

template <typename T>
auto make_cuda_pitched_malloc(size_t width, size_t height) {
    size_t pitch = 0;
    T *obj = nullptr;
    if (cudaMallocPitch(&obj, &pitch, width * sizeof(T), height) != cudaSuccess
        || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](T *ptr) { cudaFree(ptr); };
    return std::make_pair(std::unique_ptr<T[], decltype(deleter)>{obj, deleter},
                          pitch / sizeof(T));
}

class CudaEvent {
    cudaEvent_t event;

  public:
    CudaEvent() {
        if (cudaEventCreate(&event) != cudaSuccess) {
            cuda_throw_error();
        }
    }
    CudaEvent(cudaEvent_t event) : event(event) {}

    ~CudaEvent() {
        cudaEventDestroy(event);
    }
    void record(cudaStream_t stream = 0) {
        if (cudaEventRecord(event, stream) != cudaSuccess) {
            cuda_throw_error();
        }
    }
    /// Elapsed Event time, in milliseconds
    float elapsed_time(CudaEvent &since) {
        float elapsed_time = 0.0f;
        if (cudaEventElapsedTime(&elapsed_time, since.event, event) != cudaSuccess) {
            cuda_throw_error();
        }
        return elapsed_time;
    }
    void synchronize() {
        if (cudaEventSynchronize(event) != cudaSuccess) {
            cuda_throw_error();
        }
    }
};

#endif