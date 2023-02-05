#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <fmt/core.h>

#include <argparse/argparse.hpp>
#include <stdexcept>
#include <string>

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

/// Raise an exception IF CUDA is in an error state, with the name and description
auto cuda_throw_error() -> void {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_name = cudaGetErrorName(err);
        const char *err_str = cudaGetErrorString(err);
        std::string s =
          fmt::format("{}: {}\n", std::string{err_name}, std::string{err_str});
        throw cuda_error(s);
    }
}

template <class T>
class CUDAArgumentParser;

struct CUDAArguments {
  public:
    bool verbose = false;
    std::string file;

    int device_index = 0;

    cudaDeviceProp device;

  private:
    template <class T>
    friend class CUDAArgumentParser;
};

template <class ARGS = CUDAArguments>
class CUDAArgumentParser : public argparse::ArgumentParser {
    static_assert(std::is_base_of<ARGS, CUDAArguments>::value,
                  "Must be templated against a subclass of FPGAArgument");

  public:
    typedef ARGS ArgumentType;

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
              cudaGetDeviceCount(&deviceCount);

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

    auto parse_args(int argc, char **argv) -> ARGS {
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
        cudaGetDeviceProperties(&_arguments.device, _arguments.device_index);
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
    ARGS _arguments{};
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
        size_t maxw = fmt::formatted_size("{}", col);
        for (int row = slow; row < min(slow + height, data_height); ++row) {
            auto val = data[col + data_width * row];
            auto fmt_spec = fmt::format("{{:{}}}", format_type);
            maxw = std::max(maxw, fmt::formatted_size(fmt_spec, val));
            accum = max(accum, val);
        }
        col_widths.push_back(maxw);
    }

    // Draw a row header
    fmt::print("x =       ");
    for (int i = 0; i < width; ++i) {
        auto x = i + fast;
        fmt::print("{:{}} ", x, col_widths[i]);
    }
    fmt::print("\n         ┌");
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < col_widths[i]; ++j) {
            fmt::print("─");
        }
        fmt::print("─");
    }
    fmt::print("┐\n");

    for (int y = slow; y < min(slow + height, data_height); ++y) {
        if (y == slow) {
            fmt::print("y = {:2d} │", y);
        } else {
            fmt::print("    {:2d} │", y);
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
            if (dat < 0) {
                color = 9;
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
        fmt::print("\033[0m│\n");
    }
}

#endif