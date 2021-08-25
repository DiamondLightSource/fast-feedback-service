cmake_minimum_required(VERSION 3.19)

project(h5read LANGUAGES C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules")
include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

set(HDF5_FIND_QUIETLY Yes)
find_package(HDF5 REQUIRED)

add_library(h5read h5read.c)
target_include_directories(h5read PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_link_libraries(h5read PUBLIC hdf5::hdf5)

# Only build the example reader as a root script
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_executable(read_h5read read_h5read.c)
  target_link_libraries(read_h5read PUBLIC h5read)
endif()
