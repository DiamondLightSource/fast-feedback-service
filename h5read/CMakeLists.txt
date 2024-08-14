cmake_minimum_required(VERSION 3.20)
# Constraints on minimum version:
# - FindHDF5 only creates interface targets on 3.20+

project(h5read LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules")
include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

find_package(HDF5)

add_library(h5read src/h5read.c src/h5read.cc)
target_include_directories(h5read PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_link_libraries(h5read PUBLIC $<TARGET_NAME_IF_EXISTS:hdf5::hdf5>)

if (TARGET hdf5::hdf5)
  add_compile_definitions(HAVE_HDF5)
  # Problem:
  #
  #   The HDF5 Library build configuration from h5cc sometimes adds NDEBUG to
  #   INTERFACE_COMPILE_DEFINITIONS. This effectively turns off asserts for all
  #   projects in the dependency chain, even if the build_type is Debug. So,
  #   let's manually remove it, if present.
  get_target_property(_hdf5_interface_defs hdf5::hdf5 INTERFACE_COMPILE_DEFINITIONS)
  list(REMOVE_ITEM _hdf5_interface_defs NDEBUG)
  set_target_properties(hdf5::hdf5 PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${_hdf5_interface_defs}")
endif()

# Only build the example reader as a root script
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(read_h5read src/read_h5read.c)
    target_link_libraries(read_h5read PUBLIC h5read)

    add_executable(read_h5read_cpp src/read_h5read.cc)
    target_link_libraries(read_h5read_cpp PUBLIC h5read)

    add_executable(read_chunks src/read_chunks.c)
    target_link_libraries(read_chunks PUBLIC h5read)

    find_package(LZ4)
    find_package(Bitshuffle)

    if (LZ4_FOUND AND Bitshuffle_FOUND)
        add_executable(read_chunks_cpp src/read_chunks.cc)
        target_link_libraries(read_chunks_cpp PUBLIC h5read LZ4::LZ4 Bitshuffle::bitshuffle)
    endif()
endif()
