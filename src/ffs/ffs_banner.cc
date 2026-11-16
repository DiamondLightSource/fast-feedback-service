/**
 * @file ffs_banner.cc
 * @brief Print the startup banner with the installed version, for the
 *        modulefile to show when the module is loaded.
 *
 * With no arguments it prints only when stdout is a terminal, so a module load
 * inside a batch job or a service recipe stays silent. The modulefile attaches
 * stdout to its own stderr, since a modulefile's stdout is evaluated by the
 * shell as commands.
 *
 * --preview prints every colour map on both backgrounds instead, labelled
 * with the FFS_BANNER_PALETTE index that pins it.
 */
#include <unistd.h>

#include <argparse/argparse.hpp>
#include <cstdio>
#include <exception>
#include <string>

#include "banner.hpp"
#include "version.hpp"

int main(int argc, char **argv) {
    // --help and --version come with the parser
    argparse::ArgumentParser parser("ffs_banner", std::string(ffs_version()));
    parser.add_argument("--preview")
      .help("Print the banner in every palette, on dark and light backgrounds")
      .default_value(false)
      .implicit_value(true);
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }

    // Asked for explicitly, so printed even when piped
    if (parser.get<bool>("--preview")) {
        print_banner_preview("fast-feedback-service");
        return 0;
    }

    // Interactive loads only
    if (isatty(fileno(stdout))) {
        print_banner("fast-feedback-service");
    }
    return 0;
}
