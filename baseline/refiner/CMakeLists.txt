project(indexer CXX)

set(CMAKE_CXX_STANDARD 20)

find_package (Threads REQUIRED)

# Declared in cmake/Modules/Dependencies.cmake
FetchContent_MakeAvailable(mdspan)

include_directories(${CMAKE_SOURCE_DIR}/baseline/predictor)
add_subdirectory(tests)
