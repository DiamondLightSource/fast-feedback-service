project(predictor CXX)

# Automatic Dependencies
find_package(gemmi CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(baseline_predictor
    predict_cli.cc
)
target_link_libraries(baseline_predictor
    PRIVATE
    fmt
    dx2
    argparse
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    version
    ffs_common
    Eigen3::Eigen
    gemmi::gemmi_cpp
)
