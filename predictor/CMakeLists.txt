project(predictor CXX)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(Eigen3)

find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)
find_package(gemmi CONFIG REQUIRED)
# find_package(CUDAToolkit REQUIRED)

add_executable(predictor
    predict.cc
)
target_link_libraries(predictor
    PRIVATE
    fmt
    h5read
    argparse
    standalone
    LZ4::LZ4
    Bitshuffle::bitshuffle
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    dx2
    version
    ffs_common
    Eigen3::Eigen
    gemmi::gemmi_cpp
)
