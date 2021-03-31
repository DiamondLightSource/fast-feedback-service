cmake_minimum_required(VERSION 3.19)

project(baseline)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_STANDARD 11)

find_package(Boost REQUIRED)
find_package(Dials REQUIRED)


add_executable(baseline baseline.cpp local.cpp)
target_link_libraries(baseline PUBLIC Dials::Dials)