include(GoogleTest)

add_executable(test_ssx_integrate
    test_ssx_integrate.cc)
#    ${CMAKE_SOURCE_DIR}/src/integrator/sigma_estimation.cc
#    ${CMAKE_SOURCE_DIR}/baseline/integrator/ssx/ssx_integrate.cc)
target_include_directories(test_ssx_integrate PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_ssx_integrate
GTest::gtest_main dx2 Eigen3::Eigen nlohmann_json::nlohmann_json spdlog::spdlog
    ffs_common ssx_integrator)

add_executable(test_ellipsoid_parameterisation test_ellipsoid_parameterisation.cc)
target_include_directories(test_ellipsoid_parameterisation PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_ellipsoid_parameterisation
GTest::gtest_main dx2 Eigen3::Eigen ssx_integrator)

gtest_discover_tests(test_ssx_integrate PROPERTIES LABELS baseline-integrator-tests)

gtest_discover_tests(test_ellipsoid_parameterisation PROPERTIES LABELS baseline-integrator-tests)
