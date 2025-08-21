project(spotfinder CXX CUDA)

find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)
find_package(lodepng)

add_executable(spotfinder
    spotfinder.cc
    spotfinder.cu
    shmread.cc
    cbfread.cc
    kernels/masking.cu
    kernels/thresholding.cu
    kernels/erosion.cu
    connected_components/connected_components.cc
)
target_link_libraries(spotfinder
    PRIVATE
    fmt
    h5read
    argparse
    standalone
    LZ4::LZ4
    Bitshuffle::bitshuffle
    CUDA::cudart
    Boost::graph
    lodepng
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    dx2
    version
    ffs_common
    cuda_parser
)

set_property(TARGET spotfinder PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(spotfinder PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(spotfinder PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:--generate-line-info>")
