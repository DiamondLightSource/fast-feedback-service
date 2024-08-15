# Downloads lodepng and sets up targets

include(FetchContent)
# If we can't find dependencies already in the environment, fetch them
FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng
)

FetchContent_MakeAvailable(lodepng)

add_library(lodepng STATIC "${lodepng_SOURCE_DIR}/lodepng.cpp")
target_include_directories(lodepng INTERFACE "${lodepng_SOURCE_DIR}")
target_compile_definitions(lodepng INTERFACE HAVE_LODEPNG)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(lodepng DEFAULT_MSG lodepng_SOURCE_DIR)
