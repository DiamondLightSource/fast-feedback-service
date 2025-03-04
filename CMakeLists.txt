cmake_minimum_required(VERSION 3.20...3.30)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+

# Get version at start so that we can set it at root level
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
include(ResolveGitVersion)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CUDA_STANDARD 20)
project(fast-feedback-service LANGUAGES CXX C VERSION ${FFS_VERSION_CMAKE})

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

include(CheckCCompilerFlag)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set (SYSTEM_IS_x86 TRUE)
    message("x86 architechture detected.")
else ()
    set (SYSTEM_IS_x86 FALSE)
    message("x86 architechture not detected.")
endif ()

include_directories(include)

# Dependency fetching
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.0.2
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(fmt)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nlohmann_json)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY  https://github.com/ndevenish/argparse
    GIT_TAG f362c4647e7b4bbfef8320040409560b5f90e9e0
)
FetchContent_MakeAvailable(argparse)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG        v1.15.0
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(spdlog)

# Handle inclusion of dx2 dependency using git submodule
set(dx2_SOURCE_DIR "${CMAKE_SOURCE_DIR}/dx2")

if(EXISTS ${dx2_SOURCE_DIR}/CMakeLists.txt)
    add_subdirectory(${dx2_SOURCE_DIR})
    message(STATUS "Found dx2: ${dx2_SOURCE_DIR}")
else()
    message(FATAL_ERROR "dx2 submodule not found. Please run 'git submodule update --init --recursive'.")
endif()

include_directories(${dx2_SOURCE_DIR}/include)

# Make a small library that we can link to to get the version
project(version CXX C)
configure_file(version.cc.in version.cc @ONLY)
add_library(version STATIC version.cc)

enable_testing()

add_subdirectory(h5read)
add_subdirectory(baseline)

if(NOT SYSTEM_IS_x86)
    include_directories(bitshuffle/src bitshuffle/lz4)
    add_subdirectory(bitshuffle/zstd/build/cmake)
    add_library(bitshuffle
                bitshuffle/src/bitshuffle.c
                bitshuffle/lz4/lz4.c
                bitshuffle/src/bitshuffle_core.c
                bitshuffle/src/iochain.c)
endif()
