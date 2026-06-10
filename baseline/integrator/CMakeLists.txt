# baseline/integrator CMakeLists.txt
project(baseline_integrator CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(baseline_integrator
    integrator.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/extent.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/lp_correction.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/coordinate_system.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/background.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/kabsch.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/sigma_estimation.cc
)

target_link_libraries(baseline_integrator
    PRIVATE
    predictor
    fmt::fmt
    dx2
    h5read
    Eigen3::Eigen
    argparse
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ffs_common
    version
    ${CMAKE_THREAD_LIBS_INIT}
)