project(indexer CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(indexer
    indexer.cc
)
target_link_libraries(indexer
    PRIVATE
    fmt
    h5read
    argparse
    standalone
    CUDA::cudart
    CUDA::nppif
    lodepng
)
target_compile_options(indexer PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
