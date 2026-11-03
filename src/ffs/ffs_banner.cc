/**
 * @file ffs_banner.cc
 * @brief Print the startup banner with the installed version, for the
 *        modulefile to show when the module is loaded.
 *
 * Prints only when stdout is a terminal, so a module load inside a batch job
 * or a service recipe stays silent. The modulefile attaches stdout to its own
 * stderr, since a modulefile's stdout is evaluated by the shell as commands.
 */
#include <unistd.h>

#include <cstdio>

#include "banner.hpp"

int main() {
    // Interactive loads only
    if (isatty(fileno(stdout))) {
        print_banner("fast-feedback-service");
    }
    return 0;
}
