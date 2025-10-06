/**
 * @file arg_parser.hpp
 * @brief Base argument parser class and structures for Fast Feedback
 * Service applications.
 *
 * This file defines the foundational argument parsing infrastructure
 * used across all FFS applications. It provides a unified interface for
 * handling common command-line arguments such as verbose output, file
 * paths, and HDF5-specific options. The base parser can be extended by
 * derived classes to add application-specific arguments while
 * maintaining consistent parsing behavior.
 *
 * The parser supports loading additional arguments from a 'common.args'
 * file, automatic HDF5 error suppression based on verbosity settings,
 * and post-parsing hooks for custom validation and setup.
 */
#pragma once

#include <argparse/argparse.hpp>
#include <optional>
#include <string>

/**
 * @brief Structure containing parsed command-line arguments for the
 * Fast Feedback Service.
 *
 * This structure holds the common arguments that can be parsed from the
 * command line across different FFS applications. It provides a unified
 * interface for accessing parsed argument values.
 */
struct FFSArguments {
    bool verbose = false;                ///< Enable verbose logging output
    std::optional<size_t> image_number;  ///< Optional specific image number to process
};

/**
 * @brief Base argument parser class for Fast Feedback Service
 * applications.
 *
 * This class extends argparse::ArgumentParser to provide a consistent
 * interface for parsing command-line arguments across different FFS
 * applications. It handles common arguments and provides hooks for
 * derived classes to add specialized arguments while maintaining a
 * unified parsing workflow.
 */
class FFSArgumentParser : public argparse::ArgumentParser {
  public:
    explicit FFSArgumentParser(std::string version = "0.1.0");
    virtual ~FFSArgumentParser() = default;

    /**
     * @brief Adds application-specific file reading arguments to the parser.
     *
     * This pure virtual method must be implemented by derived classes to
     * define their specific file input argument patterns (e.g., single file,
     * reflection + experiment files, etc.).
     */
    virtual void add_h5read_arguments() = 0;

    /**
     * @brief Parses command-line arguments and returns structured
     * argument data.
     *
     * Processes the provided command-line arguments using the
     * configured argument definitions and returns a populated
     * FFSArguments structure. Also triggers post-parsing hooks for
     * derived classes to perform additional setup or validation.
     *
     * @param argc Number of command-line arguments
     * @param argv Array of command-line argument strings
     * @return FFSArguments Structure containing parsed argument values
     */
    auto parse_args(int argc, char **argv) -> FFSArguments;

  protected:
    /**
     * @brief Post-parsing hook for derived classes to perform
     * additional setup.
     *
     * This virtual method is called after argument parsing is complete,
     * allowing derived classes to perform validation, setup, or other
     * operations based on the parsed arguments. The base implementation
     * is empty and safe to override.
     */
    virtual void post_parse();

    FFSArguments _arguments{};       ///< Internal storage for parsed arguments
    bool _activated_h5read = false;  ///< Flag indicating if HDF5 arguments were added
};