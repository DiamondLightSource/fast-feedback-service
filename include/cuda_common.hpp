#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <fmt/core.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

// We might be on an implementation that doesn't have <span>, so use a backport
#ifdef USE_SPAN_BACKPORT
#include "span.hpp"
using tcb::span;
#else
#include <span>
using std::span;
#endif

#if __has_include(<hdf5.h>)
#define HAS_HDF5
namespace _hdf5 {
#include <hdf5.h>
}
#endif

template <typename T1, typename... TS>
auto with_formatting(const std::string &code, const T1 &first, TS... args)
  -> std::string {
    return code + fmt::format(fmt::format("{}", first), args...) + "\033[0m";
}

template <typename... T>
auto bold(T... args) -> std::string {
    return with_formatting("\033[1m", args...);
}
template <typename... T>
auto blue(T... args) -> std::string {
    return with_formatting("\033[34m", args...);
}
template <typename... T>
auto red(T... args) -> std::string {
    return with_formatting("\033[31m", args...);
}
template <typename... T>
auto green(T... args) -> std::string {
    return with_formatting("\033[32m", args...);
}
template <typename... T>
auto gray(T... args) -> std::string {
    return with_formatting("\033[37m", args...);
}
template <typename... T>
auto yellow(T... args) -> std::string {
    return with_formatting("\033[33m", args...);
}

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
        try {
            ArgumentParser::parse_args(argc, argv);
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

/// Draw a subset of the pixel values for a 2D image array
/// fast, slow, width, height - describe the bounding box to draw
/// data_width, data_height - describe the full data array size
template <typename T>
void draw_image_data(const T *data,
                     size_t fast,
                     size_t slow,
                     size_t width,
                     size_t height,
                     size_t data_width,
                     size_t data_height) {
    std::string format_type = "";
    if constexpr (std::is_integral<T>::value) {
        format_type = "d";
    } else {
        format_type = ".1f";
    }

    // Walk over the data and get various metadata for generation
    // Maximum value
    T accum = 0;
    // Maximum format width for each column
    std::vector<int> col_widths;
    for (int col = fast; col < fast + width; ++col) {
        size_t maxw = fmt::formatted_size("{:3}", col);
        for (int row = slow; row < std::min(slow + height, data_height); ++row) {
            auto val = data[col + data_width * row];
            auto fmt_spec = fmt::format("{{:{}}}", format_type);
            maxw = std::max(maxw, fmt::formatted_size(fmt_spec, val));
            accum = max(accum, val);
        }
        col_widths.push_back(maxw);
    }
    bool is_top = slow == 0;
    bool is_left = fast == 0;
    bool is_right = fast + width >= data_width;

    // Draw a row header
    fmt::print("x =       ");
    for (int i = 0; i < width; ++i) {
        auto x = i + fast;
        fmt::print("{:{}} ", x, col_widths[i]);
    }
    fmt::print("\n         ");
    if (is_top) {
        if (is_left) {
            fmt::print("╔");
        } else {
            fmt::print("╒");
        }
    } else {
        if (is_left) {
            fmt::print("╓");
        } else {
            fmt::print("┌");
        }
    }

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < col_widths[i]; ++j) {
            fmt::print(is_top ? "═" : "─");
        }
        fmt::print(is_top ? "═" : "─");
    }
    if (is_top) {
        if (is_right) {
            fmt::print("╗");
        } else {
            fmt::print("╕");
        }
    } else {
        if (is_right) {
            fmt::print("╖");
        } else {
            fmt::print("┐");
        }
    }
    fmt::print("\n");
    for (int y = slow; y < std::min(slow + height, data_height); ++y) {
        if (y == slow) {
            fmt::print("y = {:4d} {}", y, is_left ? "║" : "│");
        } else {
            fmt::print("    {:4d} {}", y, is_left ? "║" : "│");
        }
        for (int i = fast; i < fast + width; ++i) {
            // Calculate color
            // Black, 232->255, White
            // Range of 24 colors, not including white. Split into 25 bins, so
            // that we have a whole black top bin
            // float bin_scale = -25
            auto dat = data[i + data_width * y];
            int color = 255 - ((float)dat / (float)accum) * 24;
            if (color <= 231) color = 0;
            // Avoid type comparison warnings when operating on unsigned
            if constexpr (std::is_signed<decltype(dat)>::value) {
                if (dat < 0) {
                    color = 9;
                }
            }
            if (dat == accum) {
                fmt::print("\033[0m\033[1m");
            } else {
                fmt::print("\033[38;5;{}m", color);
            }
            auto fmt_spec =
              fmt::format("{{:{}{}}} ", col_widths[i - fast], format_type);
            fmt::print(fmt_spec, dat);
            if (dat == accum) {
                fmt::print("\033[0m");
            }
        }
        fmt::print("\033[0m{}\n", is_right ? "║" : "│");
    }
}
template <typename T, typename U>
void draw_image_data(const std::unique_ptr<T, U> &data,
                     size_t fast,
                     size_t slow,
                     size_t width,
                     size_t height,
                     size_t data_width,
                     size_t data_height) {
    draw_image_data(
      static_cast<T *>(data.get()), fast, slow, width, height, data_width, data_height);
}
template <typename T>
void draw_image_data(const span<const T> data,
                     size_t fast,
                     size_t slow,
                     size_t width,
                     size_t height,
                     size_t data_width,
                     size_t data_height) {
    draw_image_data(data.data(), fast, slow, width, height, data_width, data_height);
}

template <typename T>
auto make_cuda_malloc(size_t num_items = 1) {
    T *obj = nullptr;
    if (cudaMalloc(&obj, sizeof(T) * num_items) != cudaSuccess || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](T *ptr) { cudaFree(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}

template <typename T>
auto make_cuda_managed_malloc(size_t num_items) {
    T *obj = nullptr;
    if (cudaMallocManaged(&obj, sizeof(T) * num_items) != cudaSuccess
        || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](T *ptr) { cudaFree(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}
/// Allocate memory using cudaMallocHost
template <typename T>
auto make_cuda_pinned_malloc(size_t num_items = 1) {
    T *obj = nullptr;
    if (cudaMallocHost(&obj, sizeof(T) * num_items) != cudaSuccess || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](T *ptr) { cudaFreeHost(ptr); };
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

template <typename T = uint8_t>
auto GBps(float time_ms, size_t number_objects) -> float {
    return 1000 * number_objects * sizeof(T) / time_ms / 1e9;
}

template <typename T, typename U>
bool compare_results(const T *left,
                     const size_t left_pitch,
                     const U *right,
                     const size_t right_pitch,
                     std::size_t width,
                     std::size_t height,
                     size_t *mismatch_x = nullptr,
                     size_t *mismatch_y = nullptr) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            T lval = left[left_pitch * y + x];
            U rval = right[right_pitch * y + x];
            if (lval != rval) {
                if (mismatch_x != nullptr) {
                    *mismatch_x = x;
                }
                if (mismatch_y != nullptr) {
                    *mismatch_y = y;
                }
                fmt::print("First mismatch at ({}, {}), Left {} != {} Right\n",
                           x,
                           y,
                           lval,
                           rval);
                return false;
            }
        }
    }
    return true;
}

template <typename T, typename I, typename I2>
auto count_nonzero(const T *data, I width, I height, I2 pitch) -> size_t {
    size_t strong = 0;
    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            if (data[row * pitch + col]) {
                strong += 1;
            }
        }
    }
    return strong;
}
template <typename T, typename I, typename I2>
auto count_nonzero(const span<const T> data, I width, I height, I2 pitch) -> size_t {
    return count_nonzero(data.data(), width, height, pitch);
}
#endif