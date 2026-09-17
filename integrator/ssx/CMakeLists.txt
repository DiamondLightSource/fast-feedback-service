add_library(ssx_integrator STATIC
    fisher_scoring_max_likelihood.cpp
    mosaicity_parameterisation.cpp
    ssx_integrate.cpp
    max_likelihood_target.cpp
    reflection_likelihood.cpp
    ${CMAKE_SOURCE_DIR}/src/integrator/sigma_estimation.cc
)

target_include_directories(ssx_integrator PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(ssx_integrator PUBLIC
    dx2
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ffs_common
    predictor
)

nanobind_add_module(integrate
    integrate_module.cpp)

target_link_libraries(integrate PRIVATE ssx_integrator)

add_subdirectory(tests)
