include(GoogleTest)

add_executable(test_rotator test_rotator.cc)
target_link_libraries(test_rotator PRIVATE
    predictor
    GTest::gtest_main
    Eigen3::Eigen
)

gtest_discover_tests(test_rotator PROPERTIES LABELS "predictor-tests")
