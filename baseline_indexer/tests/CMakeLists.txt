include(GoogleTest)

add_executable(test_xyz_to_rlp test_xyz_to_rlp.cc)
target_include_directories(test_xyz_to_rlp PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_xyz_to_rlp GTest::gtest_main dx2 Eigen3::Eigen nlohmann_json::nlohmann_json)

add_executable(test_fft3d test_fft3d.cc)
target_include_directories(test_fft3d PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_fft3d GTest::gtest_main fmt Eigen3::Eigen pocketfft spdlog::spdlog)

add_executable(test_flood_fill test_flood_fill.cc)
target_include_directories(test_flood_fill PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_flood_fill GTest::gtest_main fmt Eigen3::Eigen spdlog::spdlog)

gtest_discover_tests(test_xyz_to_rlp PROPERTIES LABELS baseline-indexer-tests)
gtest_discover_tests(test_flood_fill PROPERTIES LABELS baseline-indexer-tests)
gtest_discover_tests(test_fft3d PROPERTIES LABELS baseline-indexer-tests)
