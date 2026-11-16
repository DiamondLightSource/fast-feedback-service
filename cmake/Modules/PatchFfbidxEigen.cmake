# Drop the Eigen version from ffbidx's own lookups, run as its patch step.
#
# ffbidx asks for Eigen 3.4 and 3, and Eigen's version file only accepts a
# request from its own major, so Eigen 5 is rejected. The conda pin chooses
# Eigen for the whole tree, so the lookups only need to find it. Rerunning on
# a patched tree changes nothing.
#
# A stopgap until ffbidx accepts Eigen 5 itself; once the pinned commit
# does, this file and the PATCH_COMMAND that runs it can go.
#
#   cmake -P PatchFfbidxEigen.cmake      (from the ffbidx source directory)

foreach(file indexer/src/CMakeLists.txt python/src/CMakeLists.txt)
    file(READ ${file} text)
    string(REGEX REPLACE "find_package\\(Eigen3 [0-9.]+ NO_MODULE\\)"
        "find_package(Eigen3 NO_MODULE)" patched "${text}")
    if(NOT patched STREQUAL text)
        file(WRITE ${file} "${patched}")
    endif()
endforeach()
