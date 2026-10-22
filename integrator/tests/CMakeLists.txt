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

add_executable(test_background_pool test_background_pool.cu ../background.cu)
target_include_directories(test_background_pool PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(test_background_pool PRIVATE
    ffs_common
    dx2
    GTest::gtest_main
    Eigen3::Eigen
    spdlog::spdlog
    CUDA::cudart
    lodepng
)

set_source_files_properties(../kabsch.cu ../background.cu PROPERTIES LANGUAGE CUDA)

function(add_test_kabsch_variant target)
    add_executable(${target} test_kabsch.cc ../kabsch.cu ../background.cu
        ${CMAKE_SOURCE_DIR}/src/integrator/extent.cc)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    target_link_libraries(${target} PRIVATE
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
    target_compile_definitions(${target} PRIVATE ${ARGN})
    set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endfunction()

add_test_kabsch_variant(test_kabsch)
# The 32-bit variant covers the widened accumulator and its 64-bit atomicAdd.
# Only the tests that build their own image can run: the rest read 16-bit
# frames, which this width cannot.
add_test_kabsch_variant(test_kabsch32 PIXEL_DATA_32BIT)

gtest_discover_tests(test_extent PROPERTIES LABELS "integrator-tests")
gtest_discover_tests(test_background PROPERTIES LABELS "integrator-tests")
gtest_discover_tests(test_background_pool PROPERTIES LABELS "integrator-tests")
gtest_discover_tests(test_kabsch PROPERTIES LABELS "integrator-tests")
gtest_discover_tests(test_kabsch32
    TEST_FILTER "*ForegroundSumSaturation*"
    TEST_PREFIX "Pixel32."
    PROPERTIES LABELS "integrator-tests")
