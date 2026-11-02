project(indexer CXX)

set(CMAKE_CXX_STANDARD 20)

find_package (Threads REQUIRED)

# Declared in cmake/Modules/Dependencies.cmake
FetchContent_MakeAvailable(pocketfft)

FetchContent_GetProperties(pocketfft)
add_library(pocketfft INTERFACE)
target_include_directories(pocketfft INTERFACE ${pocketfft_SOURCE_DIR})

add_subdirectory(tests)
add_executable(baseline_indexer
    indexer.cc
)
target_include_directories(baseline_indexer PRIVATE ${CMAKE_SOURCE_DIR}/baseline/predictor)
target_include_directories(baseline_indexer PRIVATE ${CMAKE_SOURCE_DIR}/baseline/refiner)
target_link_libraries(baseline_indexer
    PRIVATE
    fmt::fmt
    dx2
    Eigen3::Eigen
    pocketfft
    argparse
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ffs_common
    ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS baseline_indexer RUNTIME DESTINATION ${FFS_INSTALL_BINDIR} COMPONENT ffs)

# Create Python module
nanobind_add_module(index index_module.cpp)
target_include_directories(index PRIVATE ${CMAKE_SOURCE_DIR}/baseline/predictor)
target_link_libraries(index PRIVATE Eigen3::Eigen dx2 Python::Module)

install(TARGETS index LIBRARY DESTINATION ${FFS_INSTALL_PYTHONDIR} COMPONENT ffs)
