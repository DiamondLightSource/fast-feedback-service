/**
  * @file integrator.cc
 */

#include "integrator.cuh"

#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <iostream>
#include <string>

#include "common.hpp"
#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "version.hpp"

auto create_argument_parser(const std::string_view &version) -> CUDAArgumentParser {
    auto parser = CUDAArgumentParser(std::string{version});
    parser.add_h5read_arguments();
    parser.add_argument("-t", "--timeout")
      .help("Amount of time (in seconds) to wait for new images before failing.")
      .metavar("S")
      .default_value<float>(30.0f)
      .scan<'f', float>();

    return parser;
}

#pragma region Application Entry
int main(int argc, char **argv) {
    logger.info("Version: {}", FFS_VERSION);

    // Parse arguments
    auto parser = create_argument_parser(FFS_VERSION);
    auto args = parser.parse_args(argc, argv);
    float wait_timeout = parser.get<float>("timeout");

    if (!std::filesystem::exists(args.file)) {
        logger.error("File not found: {}", args.file);
        return 1;
    }
    // wait_for_ready_for_read(
    //     args.file,
    //     [](const std::string &s) { return std::filesystem::exists(s); },
    //     wait_timeout
    // );
    logger.info("Loading data from file: {}", args.file);
    ReflectionTable reflections(args.file);

    auto column_names = reflections.get_column_names();
    std::string column_names_str;
    for (const auto &name : column_names) {
        column_names_str += "\n\t- " + name;
    }
    logger.info("Column names: {}", column_names_str);

    return 0;
}
#pragma endregion Application Entry