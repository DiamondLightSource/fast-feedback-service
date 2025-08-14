# baseline/integrator CMakeLists.txt
project(baseline_integrator CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(baseline_integrator
    integrator.cc
)

target_link_libraries(baseline_integrator
    PRIVATE
    fmt
    dx2
    Eigen3::Eigen
    argparse
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ffs_common
    version
    ${CMAKE_THREAD_LIBS_INIT}
)