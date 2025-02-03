project(indexer CXX)

set(CMAKE_CXX_STANDARD 20)

find_package (Threads)

# Automatic Dependencies
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft
    GIT_TAG cpp
)
FetchContent_MakeAvailable(Eigen3)
FetchContent_MakeAvailable(pocketfft)

FetchContent_GetProperties(pocketfft)
add_library(pocketfft INTERFACE)
target_include_directories(pocketfft INTERFACE ${pocketfft_SOURCE_DIR})

# GTest could have been made available under a different name
if(NOT TARGET GTest::gtest_main)
    FetchContent_MakeAvailable(GTest)
endif()
enable_testing()

add_executable(baseline_indexer
    indexer.cc
)
target_link_libraries(baseline_indexer
    PRIVATE
    fmt
    dx2
    Eigen3::Eigen
    pocketfft
    argparse
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${CMAKE_THREAD_LIBS_INIT}
)
add_executable(test_xyz_to_rlp test_xyz_to_rlp.cc)
target_link_libraries(
    test_xyz_to_rlp GTest::gtest_main dx2 Eigen3::Eigen nlohmann_json::nlohmann_json)
add_executable(test_fft3d test_fft3d.cc)
target_link_libraries(
    test_fft3d GTest::gtest_main fmt Eigen3::Eigen pocketfft spdlog::spdlog)
include(GoogleTest)
gtest_discover_tests(test_xyz_to_rlp test_fft3d PROPERTIES LABELS baseline-indexer-tests)