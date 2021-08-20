cmake_minimum_required(VERSION 3.19)

project(baseline)

# Set the default build type to RelWithDebInfo
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Always use coloured output, unless turned off
option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${FORCE_COLORED_OUTPUT})
  add_compile_options(
    $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>
    $<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>
    $<$<CXX_COMPILER_ID:AppleClang>:-fcolor-diagnostics>
  )
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

find_package(Boost REQUIRED)
find_package(Dials REQUIRED)
find_package(HDF5 REQUIRED)
find_package(benchmark REQUIRED)

add_subdirectory(../h5read h5read)

add_library(baseline SHARED baseline.cpp )
target_link_libraries(baseline PUBLIC Dials::Dials benchmark::benchmark h5read)

add_executable(bm bm.cpp)
target_link_libraries(bm baseline h5read)

# add_executable(h5read h5read.c miniapp.c)
# target_link_libraries(h5read PUBLIC hdf5::hdf5 baseline)
