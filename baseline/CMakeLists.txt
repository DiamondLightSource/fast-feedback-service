cmake_minimum_required(VERSION 3.19)

project(baseline)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 99)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

find_package(Dials)

include_directories(../common/include)

add_subdirectory(../h5read h5read)
add_subdirectory(../dependencies/fmt fmt)

if (Dials_FOUND)
    add_library(baseline_dials SHARED baseline.cpp )
    target_link_libraries(baseline_dials PUBLIC Dials::Dials h5read)
endif()

add_library(standalone SHARED standalone.cc )
target_link_libraries(standalone PUBLIC h5read)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    find_package(benchmark REQUIRED)

    add_executable(bm bm.cpp)
    target_link_libraries(bm benchmark::benchmark $<TARGET_NAME_IF_EXISTS:baseline_dials> standalone h5read)

    
    # We can only build the comparison dials/no_tbx if we have dials
    if (Dials_FOUND)
        add_executable(miniapp miniapp.cc)
        target_link_libraries(miniapp PUBLIC baseline_dials h5read)

        add_executable(check_no_tbx check_no_tbx.cc)
        target_link_libraries(check_no_tbx PUBLIC baseline_dials h5read standalone fmt)
    endif()
endif()