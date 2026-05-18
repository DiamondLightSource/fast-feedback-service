# baseline/integrator CMakeLists.txt
project(baseline_integrator CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(baseline_integrator
    integrator.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/extent.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/lp_correction.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/coordinate_system.cc
)

target_include_directories(baseline_integrator PRIVATE ${CMAKE_SOURCE_DIR}/baseline/predictor)

target_link_libraries(baseline_integrator
    PRIVATE
    fmt
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