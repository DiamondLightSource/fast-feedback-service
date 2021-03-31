cmake_minimum_required(VERSION 3.19)

project(baseline)


# Default to release with debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_STANDARD 11)

find_package(Boost REQUIRED)
find_package(Dials REQUIRED)
find_package(HDF5 REQUIRED)
find_package(benchmark REQUIRED)

add_executable(baseline baseline.cpp )
target_link_libraries(baseline PUBLIC Dials::Dials benchmark::benchmark)

add_executable(h5read h5read.c)
target_link_libraries(h5read PUBLIC hdf5::hdf5)