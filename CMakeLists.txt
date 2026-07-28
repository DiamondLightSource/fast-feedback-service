cmake_minimum_required(VERSION 3.20...3.30)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+

# Get version at start so that we can set it at root level
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
include(ResolveGitVersion)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Set the CUDA compute capability target(s). Defaults to autodetecting
# the GPU in the build machine and building for that
include(SetCudaArchitectures)

project(fast-feedback-service LANGUAGES CXX VERSION ${FFS_VERSION_CMAKE})

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Precision for CUDA calculations. Double is the default because it is
# bit-exact against the CPU baseline; the reduced-precision (single) path
# is faster on FP64-throttled GPUs but drifts the pixel classification off
# the double result on ~1% of reflections (intensity impact stays within
# noise, but parity is lost). Opt into it deliberately.
option(USE_REDUCED_PRECISION "Use reduced (single) precision for CUDA calculations - faster but drifts pixel classification off the bit-exact double result" OFF)

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

# Discover GoogleTest cases at ctest time, not build time. The default
# POST_BUILD mode runs each test binary during the build, where conda's
# rpath makes it load the stale installed libdx2.so and fail on any dx2
# ABI change. gtest_discover_tests() reads this default, so it also
# covers the dx2 submodule without editing it. Must be set before
# add_subdirectory(dx2).
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

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

# Eigen's device std::complex helpers need constexpr as __host__ __device__ (#20015).
target_compile_options(compile_options INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)

# Add compile definitions if enabled
if(USE_REDUCED_PRECISION)
    target_compile_definitions(compile_options INTERFACE USE_REDUCED_PRECISION)
    message(STATUS "Building with reduced (single) precision")
else()
    message(STATUS "Building with double precision")
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
    spdlog::spdlog
    fmt::fmt
    argparse
)

# Check if CUDA is available
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA found: Building CUDA components.")
    set(FFS_ENABLE_CUDA ON)
    
    # SetCudaArchitectures already probed the local GPU; check that what
    # we are about to compile can actually run on it.
    if(FFS_DETECTED_GPU_ARCH)
        # A cubin loads on any later minor revision of its own major
        # generation but never across one (an sm_80 cubin runs on 8.6,
        # 8.7 and 8.9), and the driver can JIT PTX onto any device at or
        # above its virtual architecture. Warn only when neither route
        # covers the card that is actually here.
        set(GPU_IS_COVERED FALSE)
        math(EXPR DETECTED_MAJOR "${FFS_DETECTED_GPU_ARCH} / 10")
        foreach(arch IN LISTS FFS_CUDA_ARCH_REAL)
            math(EXPR ARCH_MAJOR "${arch} / 10")
            if(ARCH_MAJOR EQUAL DETECTED_MAJOR AND NOT arch GREATER FFS_DETECTED_GPU_ARCH)
                set(GPU_IS_COVERED TRUE)
            endif()
        endforeach()
        foreach(arch IN LISTS FFS_CUDA_ARCH_VIRTUAL)
            if(NOT arch GREATER FFS_DETECTED_GPU_ARCH)
                set(GPU_IS_COVERED TRUE)
            endif()
        endforeach()

        if(GPU_IS_COVERED)
            message(STATUS "✓ Compiled architectures cover the detected GPU")
        else()
            message(WARNING
                "Nothing being compiled can run on the GPU in this machine (sm_${FFS_DETECTED_GPU_ARCH}).\n"
                "   Building for: ${CMAKE_CUDA_ARCHITECTURES}\n"
                "   The resulting binaries will fail to load here.\n"
                "   To build for the detected GPU: cmake -DCUDA_ARCH=native ..\n"
                "   To build what the container ships: cmake -DCUDA_ARCH=all-supported .."
            )
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
