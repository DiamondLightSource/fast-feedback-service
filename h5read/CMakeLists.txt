cmake_minimum_required(VERSION 3.19)

project(h5read LANGUAGES C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules")
include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

include(CheckIncludeFileCXX)

set(HDF5_FIND_QUIETLY Yes)
find_package(HDF5 REQUIRED)

add_library(h5read src/h5read.c src/h5read.cc)

target_include_directories(h5read PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_link_libraries(h5read PUBLIC hdf5::hdf5)

# Check if we have the C++20 span header. Otherwise, use the backport
check_include_file_cxx(span HAS_CXX20_SPAN)
if(NOT HAS_CXX20_SPAN)
  # We don't want to build the projects testing, so just recreate the interface here
  add_library(span INTERFACE)
  target_sources(span INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/span/include/tcb/span.hpp)
  target_include_directories(span INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/span/include/tcb)
  target_compile_features(span INTERFACE cxx_std_11)
  target_link_libraries(h5read PUBLIC span)
  target_compile_definitions(h5read PUBLIC USE_SPAN_BACKPORT)
endif()

# Only build the example reader as a root script
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_executable(read_h5read src/read_h5read.c)
  target_link_libraries(read_h5read PUBLIC h5read)

  add_executable(read_h5read_cpp src/read_h5read.cc)
  target_link_libraries(read_h5read_cpp PUBLIC h5read)
endif()
