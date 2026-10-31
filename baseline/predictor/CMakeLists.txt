project(predictor CXX)

# Automatic Dependencies
find_package(gemmi CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

# Spot prediction compiled once and linked, rather than source-included.
add_library(predictor STATIC
    ${CMAKE_SOURCE_DIR}/src/predictor/utils.cc
    ${CMAKE_SOURCE_DIR}/src/predictor/ray_predictors.cc
    ${CMAKE_SOURCE_DIR}/src/predictor/threadpool.cc
    ${CMAKE_SOURCE_DIR}/src/predictor/predict.cc
)
target_include_directories(predictor PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(predictor
    PUBLIC
    fmt::fmt
    dx2
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ffs_common
    Eigen3::Eigen
    gemmi::gemmi_cpp
)

add_executable(baseline_predictor
    predict_cli.cc
)
target_link_libraries(baseline_predictor
    PRIVATE
    predictor
    fmt::fmt
    dx2
    argparse
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    version
    ffs_common
    Eigen3::Eigen
    gemmi::gemmi_cpp
)
