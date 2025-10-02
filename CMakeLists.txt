cmake_minimum_required(VERSION 3.20...3.30)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+

# Get version at start so that we can set it at root level
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
include(ResolveGitVersion)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
project(fast-feedback-service LANGUAGES CXX VERSION ${FFS_VERSION_CMAKE})

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Option to use 32-bit pixel data instead of 16-bit
option(PIXEL_DATA_32BIT "Use 32-bit pixel data instead of 16-bit" OFF)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

include_directories(include)

# Dependency fetching
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nlohmann_json)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY  https://github.com/ndevenish/argparse
    GIT_TAG f362c4647e7b4bbfef8320040409560b5f90e9e0
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(argparse)

find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

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
project(version CXX)
configure_file(version.cc.in version.cc @ONLY)
add_library(version STATIC version.cc)

# Create a header-only interface library for compile definitions
add_library(compile_options INTERFACE)

# Add compile definitions if enabled
if(PIXEL_DATA_32BIT)
    target_compile_definitions(compile_options INTERFACE PIXEL_DATA_32BIT)
    message(STATUS "Building with 32-bit pixel data support")
else()
    message(STATUS "Building with 16-bit pixel data support")
endif()

# Make a common C++ library for shared code
add_library(ffs_common STATIC 
    src/ffs/ffs_logger.cc
    src/ffs/arg_parser.cc
)
target_include_directories(ffs_common PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(ffs_common PUBLIC
    compile_options
    spdlog
    fmt
    argparse
)

# Check if CUDA is available
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA found: Building CUDA components.")
    set(FFS_ENABLE_CUDA ON)
else()
    message(STATUS "CUDA not found: Skipping CUDA components.")
    set(FFS_ENABLE_CUDA OFF)
endif()

# Conditionally add a CUDA common library
if (FFS_ENABLE_CUDA)
    add_library(cuda_parser STATIC
        src/ffs/cuda_arg_parser.cc
    )

    target_include_directories(cuda_parser PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )

    target_link_libraries(cuda_parser
        ffs_common
        argparse
        CUDA::cudart
    )
endif()

enable_testing()

add_subdirectory(h5read)
add_subdirectory(baseline)
add_subdirectory(spotfinder)
