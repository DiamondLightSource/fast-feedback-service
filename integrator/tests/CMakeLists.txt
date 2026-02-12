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

add_executable(test_kabsch test_kabsch.cc ../kabsch.cu)
set_source_files_properties(../kabsch.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_kabsch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
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
gtest_discover_tests(test_kabsch PROPERTIES LABELS "integrator-tests")