include(GoogleTest)

add_executable(test_connected_components
    test_connected_components.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../connected_components/connected_components.cc
)
target_include_directories(test_connected_components PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../connected_components
)
target_link_libraries(test_connected_components PRIVATE
    ffs_common
    dx2
    h5read
    Boost::graph
    GTest::gtest_main
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    CUDA::cudart
    lodepng
)

gtest_discover_tests(test_connected_components PROPERTIES LABELS "spotfinder-tests")
