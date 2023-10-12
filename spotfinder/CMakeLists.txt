cmake_minimum_required(VERSION 3.20)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+

project(spotfinder CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

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

add_library(lodepng STATIC ../../dependencies/lodepng/lodepng.cpp)
target_include_directories(lodepng INTERFACE ../../dependencies/lodepng)
target_compile_definitions(lodepng INTERFACE HAVE_LODEPNG)

add_library(json INTERFACE)
target_include_directories(json INTERFACE ../../dependencies/json/include)

find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)

add_executable(spotfinder spotfinder.cc spotfinder.cu shmread.cc cbfread.cc)
target_link_libraries(spotfinder
    PRIVATE
    fmt
    h5read
    argparse
    standalone
    LZ4::LZ4
    Bitshuffle::bitshuffle
    CUDA::cudart
    CUDA::nppif
    Boost::graph
    lodepng
    json
)
target_compile_options(spotfinder PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")

