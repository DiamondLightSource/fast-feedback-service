project(integrator CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)

function(add_integrator_variant target)
    add_executable(${target}
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

    target_include_directories(${target}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../spotfinder
    )

    target_link_libraries(${target}
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

    target_compile_definitions(${target} PRIVATE ${ARGN})

    # Allow CUDA to use multi-file device variables
    set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${target} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
    target_compile_options(${target} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:--generate-line-info>")

    install(TARGETS ${target} RUNTIME DESTINATION ${FFS_INSTALL_BINDIR} COMPONENT ffs)
endfunction()

add_integrator_variant(integrator)
add_integrator_variant(integrator32 PIXEL_DATA_32BIT)

add_subdirectory(ssx)
add_subdirectory(tests)
