# Every dependency fetched from git, and the tag or commit it is pinned to.
#
# FetchContent keeps the first declaration of a name and ignores any later
# one, so declaring everything here, before any add_subdirectory, fixes each
# version for the whole tree, the dx2 submodule included. Consumers make a
# dependency available where they use it and never declare it themselves.
# Whatever conda-forge serves at the right version comes from the
# environment instead; nanobind and GTest carry FIND_PACKAGE_ARGS, so the
# environment's copy wins and git is only the fallback.
#
# The CI cache of fetched sources is keyed on this file, so a changed pin
# starts that cache empty. A declaration made anywhere else would be built
# from a stale cached tree.

set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

# conda-forge ships nanobind as a python package with its cmake config inside
# it, out of reach of the prefix search, so the interpreter is asked for it.
# The version is exact so the environment and the fallback agree.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.8.0
    FIND_PACKAGE_ARGS 2.8.0 EXACT CONFIG
)

# A fork of p-ranav/argparse carrying terminal-width wrapping and mutually
# exclusive groups, so it is not interchangeable with the packaged upstream.
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/ndevenish/argparse
    GIT_TAG f362c4647e7b4bbfef8320040409560b5f90e9e0
    EXCLUDE_FROM_ALL
)

# Twelve commits past v1.1.3, and conda-forge builds it for x86_64 only. The
# indexer links the Eigen3::Eigen already found in the conda environment, so
# its only submodule, Eigen on gitlab.com, is never compiled and is skipped.
FetchContent_Declare(
    ffbidx
    GIT_REPOSITORY https://github.com/paulscherrerinstitute/fast-feedback-indexer.git
    GIT_TAG 6f6e3c78d847dce6fdebf0a93dcd4440af0737f4
    GIT_SUBMODULES ""
)

# lodepng publishes no releases, and the conda-forge build dates from 2022.
FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng
    GIT_TAG e584a8490db1b76b81bf298c11c09d3b55d9c270
)

# The C++ header lives on a branch with no releases and is not packaged.
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft
    GIT_TAG 4d865cba24359b05b37da4e75aee24498c3b78e7
)

# Matches the environment's gtest, which is found first.
FetchContent_Declare(
    GTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)
