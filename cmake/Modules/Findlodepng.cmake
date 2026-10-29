# Downloads lodepng and sets up targets

include(FetchContent)
# If we can't find dependencies already in the environment, fetch them.
# lodepng publishes no releases, so it is pinned to a commit.
FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng
    GIT_TAG e584a8490db1b76b81bf298c11c09d3b55d9c270
)

FetchContent_MakeAvailable(lodepng)

add_library(lodepng STATIC "${lodepng_SOURCE_DIR}/lodepng.cpp")
target_include_directories(lodepng INTERFACE "${lodepng_SOURCE_DIR}")
target_compile_definitions(lodepng INTERFACE HAVE_LODEPNG)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(lodepng DEFAULT_MSG lodepng_SOURCE_DIR)
