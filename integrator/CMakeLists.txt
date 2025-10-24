project(integrator CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)

add_executable(integrator
    integrator.cc
    integrator.cu
    kabsch.cu
    extent.cu
    ../spotfinder/shmread.cc
    ../spotfinder/cbfread.cc
)

target_include_directories(integrator
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../spotfinder
)

target_link_libraries(integrator
    PRIVATE
    fmt
    h5read
    argparse
    standalone
    LZ4::LZ4
    Bitshuffle::bitshuffle
    CUDA::cudart
    lodepng
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    dx2
    Eigen3::Eigen
    version
    ffs_common
    cuda_parser
)

# Allow CUDA to use multi-file device variables
set_property(TARGET integrator PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(integrator PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(integrator PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:--generate-line-info>")
