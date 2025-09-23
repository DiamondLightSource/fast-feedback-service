/**
 * @file arg_parser.cc
 * @brief Implementation of the base argument parser for Fast Feedback
 * Service applications.
 *
 * This file contains the implementation of FFSArgumentParser, which
 * provides consistent command-line argument parsing across all FFS
 * applications. It handles common arguments like verbose output and
 * file paths, with special support for HDF5/Nexus file processing.
 *
 * Key features include:
 * - Automatic loading of additional arguments from 'common.args' file
 * - HDF5 error message suppression based on verbosity settings
 * - Mutually exclusive groups for sample vs file input modes
 * - Post-parsing hooks for derived class customization
 * - Formatted error reporting with usage information
 */
#include "arg_parser.hpp"

#include <fmt/core.h>

#include <argparse/argparse.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "common.hpp"

#ifdef HAS_HDF5
namespace _hdf5 {
#include <hdf5.h>
}
#endif

FFSArgumentParser::FFSArgumentParser(std::string version)
    : ArgumentParser("", version, argparse::default_arguments::help) {
    add_argument("--version")
      .help("Print version information and exit")
      .action([=](const auto &) {
          fmt::print("{}\n", version);
          std::exit(0);
      })
      .default_value(false)
      .implicit_value(true)
      .nargs(0);

    add_argument("-v", "--verbose")
      .help("Verbose output")
      .implicit_value(false)
      .action([&](const std::string &) { _arguments.verbose = true; });

    // Initialize HDF5 reading arguments by default
    add_h5read_arguments();
}

void FFSArgumentParser::add_h5read_arguments() {
    // Check if implicit sample is enable via environment variable
    bool implicit_sample = std::getenv("H5READ_IMPLICIT_SAMPLE") != nullptr;
    // Create a mutually exclusive group for sample vs file input
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
      .action([&](const std::string &val) { _arguments.file = val; });

    _activated_h5read = true;
}

auto FFSArgumentParser::parse_args(int argc, char **argv) -> FFSArguments {
    // Convert command line arguments to vector for easier manipulation
    std::vector<std::string> args{argv, argv + argc};

    // Load additional arguments from common.args file if it exists
    std::filesystem::path argfile{"common.args"};
    if (std::filesystem::exists(argfile)) {
        std::ifstream f(argfile);
        std::string arg;
        // Read each line as a separate argument
        while (std::getline(f, arg)) {
            // Only add non-empty arguments that aren't already present
            if (!arg.empty()
                && std::find(args.begin(), args.end(), arg) == args.end()) {
                args.push_back(arg);
            }
        }
    }

    try {
        ArgumentParser::parse_args(args);
    } catch (const std::runtime_error &e) {
        fmt::print("{}: {}\n{}\n", bold(red("Error")), red(e.what()), usage());
        std::exit(1);
    }

#ifdef HAS_HDF5
    // Suppress HDF5 error messages unless verbose mode is enabled
    if (_activated_h5read && !_arguments.verbose) {
        _hdf5::H5Eset_auto((_hdf5::hid_t)0, NULL, NULL);
    }
#endif

    // Call post-parsing hook for derived classes
    post_parse();

    return _arguments;
}

void FFSArgumentParser::post_parse() {
    // Default implementation does nothing
    // Derived classes override this method to perform additional setup
}