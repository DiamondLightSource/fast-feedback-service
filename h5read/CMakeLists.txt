cmake_minimum_required(VERSION 3.19)

project(h5read LANGUAGES C)

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

set(CMAKE_C_STANDARD 99)

find_package(HDF5 REQUIRED)

add_library(h5read h5read.c)
target_include_directories(h5read PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_link_libraries(h5read PUBLIC hdf5::hdf5)

add_executable(h5read_exec miniapp.c)
set_target_properties(h5read_exec PROPERTIES OUTPUT_NAME h5read)
target_link_libraries(h5read_exec PUBLIC h5read)
