include(GoogleTest)

add_executable(test_extent test_extent.cc ${CMAKE_SOURCE_DIR}/src/integrator/extent.cc)
target_include_directories(test_extent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
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

add_executable(test_kabsch test_kabsch.cc ../kabsch.cu ../background.cu ${CMAKE_SOURCE_DIR}/src/integrator/extent.cc)
set_source_files_properties(../kabsch.cu ../background.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_kabsch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(test_kabsch PRIVATE
    ffs_common
    dx2
    h5read
    GTest::gtest_main
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    CUDA::cudart
    lodepng
)
set_property(TARGET test_kabsch PROPERTY CUDA_SEPARABLE_COMPILATION ON)

gtest_discover_tests(test_extent PROPERTIES LABELS "integrator-tests")
gtest_discover_tests(test_background PROPERTIES LABELS "integrator-tests")
gtest_discover_tests(test_kabsch PROPERTIES LABELS "integrator-tests")
