include(GoogleTest)

add_executable(test_extent test_extent.cc ../extent.cc)
target_include_directories(test_extent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_extent PRIVATE 
    ffs_common
    dx2
    GTest::gtest_main
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

gtest_discover_tests(test_extent PROPERTIES LABELS "integrator")