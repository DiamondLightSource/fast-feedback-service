cmake_minimum_required(VERSION 3.19)

project(h5read LANGUAGES C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules")
include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

set(HDF5_FIND_QUIETLY Yes)
find_package(HDF5 REQUIRED)

if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 20)
  set(CMAKE_CXX_STANDARD 20)
endif()

add_library(h5read src/h5read.c src/h5read.cc)

target_include_directories(h5read PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_link_libraries(h5read PUBLIC hdf5::hdf5)

# Only build the example reader as a root script
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_executable(read_h5read src/read_h5read.c)
  target_link_libraries(read_h5read PUBLIC h5read)

  add_executable(read_h5read_cpp src/read_h5read.cc)
  target_link_libraries(read_h5read_cpp PUBLIC h5read)
endif()
