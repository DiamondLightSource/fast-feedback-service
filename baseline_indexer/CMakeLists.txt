project(indexer CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Automatic Dependencies
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS
)
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft
    GIT_TAG cpp
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(Eigen3)
FetchContent_MakeAvailable(pocketfft)

FetchContent_GetProperties(pocketfft)
add_library(pocketfft INTERFACE)
target_include_directories(pocketfft INTERFACE ${pocketfft_SOURCE_DIR})

add_executable(baseline_indexer
    indexer.cc
)
target_link_libraries(baseline_indexer
    PRIVATE
    fmt
    h5read
    dx2
    Eigen3::Eigen
    pocketfft
    argparse
    standalone
    CUDA::cudart
    CUDA::nppif
    lodepng
    nlohmann_json::nlohmann_json
)
target_compile_options(baseline_indexer PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
