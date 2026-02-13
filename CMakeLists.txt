cmake_minimum_required(VERSION 3.20...3.30)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+

# Get version at start so that we can set it at root level
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
include(ResolveGitVersion)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Set CUDA compute capability target
# Override with just the number: cmake -DCUDA_ARCH=86 ..
# Set default to 7.5 (Turing)
set(CUDA_ARCH "75 - Turing (RTX 20xx, GTX 16xx)" CACHE STRING "CUDA compute capability target architecture")
set_property(CACHE CUDA_ARCH PROPERTY STRINGS 
    "52 - Maxwell (GTX 9xx, Titan X)"
    "60 - Pascal (GTX 10xx, P100)"
    "61 - Pascal (GTX 1050/1030)"
    "70 - Volta (V100, Titan V100)"
    "75 - Turing (RTX 20xx, GTX 16xx)"
    "80 - Ampere (A100)"
    "86 - Ampere (RTX 30xx)"
    "89 - Ada Lovelace (RTX 40xx)"
    "90 - Hopper (H100)" 
)

# Extract just the number from the selection (digits before the " - " label)
string(REGEX MATCH "^[0-9]+" CUDA_ARCH_NUMBER "${CUDA_ARCH}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_NUMBER})
message(STATUS "CUDA architecture set to: sm_${CMAKE_CUDA_ARCHITECTURES}")

project(fast-feedback-service LANGUAGES CXX VERSION ${FFS_VERSION_CMAKE})

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Option to use double precision for CUDA calculations
option(USE_DOUBLE_PRECISION "Use double precision for CUDA calculations" OFF)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

include_directories(include)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Dependency fetching
# Find packages that are in conda environment
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json 3.11.3 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

# Dependency fetching - only for packages NOT in conda
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.8.0
)
FetchContent_MakeAvailable(nanobind)

FetchContent_Declare(
    argparse
    GIT_REPOSITORY  https://github.com/ndevenish/argparse
    GIT_TAG f362c4647e7b4bbfef8320040409560b5f90e9e0
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(argparse)

# Required for Python module build of ffbidx
set(BUILD_FAST_INDEXER ON CACHE BOOL "" FORCE)
set(PYTHON_MODULE ON CACHE BOOL "" FORCE)
set(PYTHON_MODULE_RPATH ON CACHE BOOL "" FORCE)

# ffbidx uses this to find site-packages
set(PYTHON_MODULE_PATH ${Python_SITEARCH} CACHE PATH "ffbidx python site-packages path" FORCE)

FetchContent_Declare(
  ffbidx
  GIT_REPOSITORY https://github.com/paulscherrerinstitute/fast-feedback-indexer.git
  GIT_TAG main
)
FetchContent_MakeAvailable(ffbidx)

# GTest - try find_package first, fallback to FetchContent
find_package(GTest QUIET)
if(NOT GTest_FOUND AND NOT TARGET GTest::gtest_main)
    FetchContent_Declare(
        GTest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(GTest)
endif()

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

# Create an interface library for compile definitions
add_library(compile_options INTERFACE)

# Add compile definitions if enabled
if(USE_DOUBLE_PRECISION)
    target_compile_definitions(compile_options INTERFACE USE_DOUBLE_PRECISION)
    message(STATUS "Building with double precision")
else()
    message(STATUS "Building with single precision")
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
    
    # Try to detect GPU compute capability
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE DETECTED_GPU_CAPS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(DETECTED_GPU_CAPS)
        # nvidia-smi returns "7.5" format, convert to "75"
        string(REPLACE "." "" DETECTED_GPU_ARCH "${DETECTED_GPU_CAPS}")
        # Get first GPU if multiple
        string(REGEX MATCH "[0-9]+" DETECTED_GPU_ARCH "${DETECTED_GPU_ARCH}")
        message(STATUS "Detected GPU compute capability: sm_${DETECTED_GPU_ARCH}")
        
        # Warn if configured architecture doesn't match detected GPU
        if(NOT CUDA_ARCH_NUMBER STREQUAL DETECTED_GPU_ARCH)
            message(WARNING 
                "Target CUDA architecture (sm_${CUDA_ARCH_NUMBER}) differs from detected GPU (sm_${DETECTED_GPU_ARCH}).\n"
                "   The compiled code may not run optimally or at all on this system.\n"
                "   To build for the detected GPU: cmake -DCUDA_ARCH=${DETECTED_GPU_ARCH} ..\n"
                "   To build for a different target: ensure the target system has sm_${CUDA_ARCH_NUMBER} or newer."
            )
        else()
            message(STATUS "✓ Configured architecture matches detected GPU")
        endif()
    else()
        message(STATUS "Could not detect GPU compute capability (nvidia-smi not available or no GPU present)")
    endif()
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
add_subdirectory(integrator)
