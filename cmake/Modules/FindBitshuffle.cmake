# Find the Bitshuffle module header and static library
#
# Creates the target:
#       Bitshuffle::Bitshuffle
#
# And sets the variables:
#       Bitshuffle_FOUND
#       BITSHUFFLE_INCLUDE
#       BITSHUFFLE_STATIC_LIBRARY

find_path(BITSHUFFLE_INCLUDE bitshuffle.h REQUIRED)
find_library(BITSHUFFLE_STATIC_LIBRARY bitshuffle.a REQUIRED)

find_package(LZ4)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Bitshuffle DEFAULT_MSG BITSHUFFLE_STATIC_LIBRARY BITSHUFFLE_INCLUDE LZ4_FOUND)

if(Bitshuffle_FOUND AND NOT TARGET Bitshuffle::bitshuffle)
    add_library(Bitshuffle::bitshuffle STATIC IMPORTED)
    set_target_properties(Bitshuffle::bitshuffle  PROPERTIES IMPORTED_LOCATION "${BITSHUFFLE_STATIC_LIBRARY}")
    target_include_directories(Bitshuffle::bitshuffle  INTERFACE "${BITSHUFFLE_INCLUDE}")
    target_link_libraries(Bitshuffle::bitshuffle INTERFACE LZ4::LZ4)
endif()