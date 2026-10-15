project(integrator CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)

add_executable(integrator
    integrator.cc
    integrator.cu
    kabsch.cu
    background.cu
    ${CMAKE_SOURCE_DIR}/src/integrator/extent.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/lp_correction.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/coordinate_system.cc
    ${CMAKE_SOURCE_DIR}/src/integrator/sigma_estimation.cc
    ../spotfinder/shmread.cc
    ../spotfinder/cbfread.cc
)

add_executable(serial_integrator
    ssx_integrator.cc
    ellipsoid.cu
    ${CMAKE_SOURCE_DIR}/src/integrator/extent.cc
    ../spotfinder/shmread.cc
    ../spotfinder/cbfread.cc
)

target_include_directories(integrator
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../spotfinder
)

target_include_directories(serial_integrator
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../spotfinder
)

target_link_libraries(integrator
    PRIVATE
    predictor
    fmt::fmt
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

target_link_libraries(serial_integrator
    PRIVATE
    predictor
    fmt::fmt
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

install(TARGETS integrator RUNTIME DESTINATION ${FFS_INSTALL_BINDIR} COMPONENT ffs)

target_compile_definitions(serial_integrator PRIVATE PIXEL_DATA_32BIT)
set_property(TARGET serial_integrator PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(serial_integrator PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(serial_integrator PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:--generate-line-info>")

install(TARGETS serial_integrator RUNTIME DESTINATION ${FFS_INSTALL_BINDIR} COMPONENT ffs)

add_subdirectory(ssx)
add_subdirectory(tests)