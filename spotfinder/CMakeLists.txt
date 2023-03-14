cmake_minimum_required(VERSION 3.20)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+

project(spotfinder CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules")

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)
include(UseSpanBackportIfNeeded)

include_directories(../common/include)

add_subdirectory(../../dependencies/fmt fmt)
add_subdirectory(../../h5read h5read)
add_subdirectory(../../dependencies/argparse argparse)
add_subdirectory(../../baseline baseline)

find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)


add_executable(spotfinder spotfinder.cu)
target_link_libraries(spotfinder PRIVATE fmt h5read argparse standalone LZ4::LZ4 Bitshuffle::bitshuffle)
target_compile_options(spotfinder PRIVATE "$<$<CONFIG:Debug>:-G>")
