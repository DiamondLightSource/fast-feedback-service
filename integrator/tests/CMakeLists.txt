include(GoogleTest)

add_executable(test_integrator test_integrator.cc ../extent.cc)
target_include_directories(test_integrator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_integrator PRIVATE 
    ffs_common
    dx2
    GTest::gtest_main
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

gtest_discover_tests(test_integrator PROPERTIES LABELS "integrator")