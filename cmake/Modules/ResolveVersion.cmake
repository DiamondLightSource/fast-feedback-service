# Resolve the project version and build the `ffs_version` library around it.
#
# setuptools_scm is the one resolver, configured in pyproject.toml;
# every other consumer reads what it produces. The compiled programs
# read ffs_version() from the generated version.cc, and the python package
# imports __version__ from the generated _version.py, so the binaries
# and the package cannot disagree.
#
# A pip build resolves the version before cmake starts and hands it over
# as SKBUILD_PROJECT_VERSION_FULL, which is taken as fixed. A plain
# cmake tree has no such handover, so it resolves the version again on
# every build: a dev binary then names the commit, and whether the tree
# was dirty, as of the build rather than as of the last configure. That
# costs one setuptools_scm run per build. The byproducts are restat by
# Ninja, so a rebuild relinks only when the version string actually
# changes.
#
# pip keeps its build directory between runs and caches the version it
# passed in. Running ninja directly in that directory keeps pip's fixed
# version, so a plain cmake build wants a build directory of its own.
#
# Include after find_package(Python): the interpreter that builds the
# extension modules is the one that runs setuptools_scm.
#
# Sets FFS_VERSION_FULL, a PEP 440 version string.

include_guard(GLOBAL)

if(NOT Python_EXECUTABLE)
    message(FATAL_ERROR "find_package(Python) must run before including ResolveVersion")
endif()

set(FFS_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(FFS_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(FFS_VERSION_FULL "${SKBUILD_PROJECT_VERSION_FULL}")

# Resolve once now, so the version is known at configure time and the
# generated files exist before anything compiles
include(WriteVersion)

if(SKBUILD_PROJECT_VERSION_FULL)
    message(STATUS "Version from the python build: ${FFS_VERSION_FULL}")
else()
    message(STATUS "Version from setuptools_scm: ${FFS_VERSION_FULL}")

    # Resolve again on every build of a plain cmake tree
    add_custom_target(ffs_version_resolve
        COMMAND "${CMAKE_COMMAND}"
            "-DFFS_SOURCE_DIR=${FFS_SOURCE_DIR}"
            "-DFFS_BINARY_DIR=${FFS_BINARY_DIR}"
            "-DPython_EXECUTABLE=${Python_EXECUTABLE}"
            -P "${CMAKE_CURRENT_LIST_DIR}/WriteVersion.cmake"
        BYPRODUCTS "${FFS_BINARY_DIR}/version.cc" "${FFS_BINARY_DIR}/_version.py"
        COMMENT "Resolving the version"
        VERBATIM
    )
endif()

add_library(ffs_version STATIC "${FFS_BINARY_DIR}/version.cc")
if(TARGET ffs_version_resolve)
    add_dependencies(ffs_version ffs_version_resolve)
endif()

# Installed beside the extension modules, so the wheel, an editable install
# and a plain cmake install all carry it
install(FILES "${FFS_BINARY_DIR}/_version.py"
    DESTINATION ${FFS_INSTALL_PYTHONDIR}
    COMPONENT ffs
)
