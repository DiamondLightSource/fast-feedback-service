include(GoogleTest)

add_executable(test_gradients_calculator test_gradients_calculator.cc)
target_include_directories(test_gradients_calculator PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(test_gradients_calculator PRIVATE "${PROJECT_SOURCE_DIR}/baseline/indexer")
target_link_libraries(test_gradients_calculator mdspan GTest::gtest_main Eigen3::Eigen nlohmann_json::nlohmann_json)

add_executable(test_detector_parameterisation test_detector_parameterisation.cc)
target_include_directories(test_detector_parameterisation PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_detector_parameterisation mdspan GTest::gtest_main Eigen3::Eigen nlohmann_json::nlohmann_json)

gtest_discover_tests(test_gradients_calculator PROPERTIES LABELS baseline-indexer-tests)
gtest_discover_tests(test_detector_parameterisation PROPERTIES LABELS baseline-indexer-tests)
