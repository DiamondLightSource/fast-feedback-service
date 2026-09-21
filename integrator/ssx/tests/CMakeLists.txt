include(GoogleTest)

add_executable(test_ssx_integrate test_ssx_integrate.cpp)
target_include_directories(test_ssx_integrate PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_ssx_integrate
GTest::gtest_main dx2 Eigen3::Eigen nlohmann_json::nlohmann_json spdlog::spdlog
    ffs_common ssx_integrator)

add_executable(test_mosaicity_parameterisation test_mosaicity_parameterisation.cpp)
target_include_directories(test_mosaicity_parameterisation PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_mosaicity_parameterisation
GTest::gtest_main dx2 Eigen3::Eigen ssx_integrator)

find_package(gemmi CONFIG REQUIRED)

add_executable(test_index_generator test_index_generator.cpp)
target_include_directories(test_index_generator PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_index_generator
GTest::gtest_main dx2 Eigen3::Eigen gemmi::gemmi_cpp)

gtest_discover_tests(test_ssx_integrate PROPERTIES LABELS ssx-integrator-tests)

gtest_discover_tests(test_mosaicity_parameterisation PROPERTIES LABELS ssx-integrator-tests)

gtest_discover_tests(test_index_generator PROPERTIES LABELS ssx-integrator-tests)
