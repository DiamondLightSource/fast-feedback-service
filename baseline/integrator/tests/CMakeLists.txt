include(GoogleTest)

add_executable(test_extent test_extent.cc ${CMAKE_SOURCE_DIR}/src/integrator/extent.cc)
target_link_libraries(test_extent PRIVATE
    ffs_common
    dx2
    GTest::gtest_main
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

add_executable(test_background test_background.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/background.cc)
target_include_directories(test_background PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_background PRIVATE
    GTest::gtest_main
)

gtest_discover_tests(test_extent PROPERTIES LABELS baseline-integrator-tests)
gtest_discover_tests(test_background PROPERTIES LABELS baseline-integrator-tests)
