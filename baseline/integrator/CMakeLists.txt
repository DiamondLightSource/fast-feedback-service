# baseline/integrator CMakeLists.txt
project(baseline_integrator CXX)

set(CMAKE_CXX_STANDARD 20)

FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan
    GIT_TAG mdspan-0.6.0
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(mdspan)

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
    h5read
    argparse
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ffs_common
    version
    ${CMAKE_THREAD_LIBS_INIT}
)
add_subdirectory(ssx)

target_link_libraries(integrate
    PRIVATE
    fmt::fmt
    Eigen3::Eigen
    dx2
    spdlog::spdlog
    ffs_common
    Python::Module
)

# Install into the runtime Python sitearch
install(TARGETS integrate
    LIBRARY DESTINATION ${Python_SITEARCH}/ffs
    COMPONENT Runtime
)
