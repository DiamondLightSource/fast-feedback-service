cmake_minimum_required(VERSION 3.19)

project(h5read LANGUAGES C)

# If we're running as the root script, default to RelWithDebInfo
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

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
endif()

find_package(HDF5 REQUIRED)

add_library(h5read h5read.c)
target_include_directories(h5read PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_link_libraries(h5read PUBLIC hdf5::hdf5)

# Only build the example reader as a root script
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_executable(read_h5read read_h5read.c)
  target_link_libraries(read_h5read PUBLIC h5read)
endif()