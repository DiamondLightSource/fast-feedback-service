# Downloads lodepng and sets up targets

# Declared in cmake/Modules/Dependencies.cmake
FetchContent_MakeAvailable(lodepng)

add_library(lodepng STATIC "${lodepng_SOURCE_DIR}/lodepng.cpp")
target_include_directories(lodepng INTERFACE "${lodepng_SOURCE_DIR}")
target_compile_definitions(lodepng INTERFACE HAVE_LODEPNG)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(lodepng DEFAULT_MSG lodepng_SOURCE_DIR)
