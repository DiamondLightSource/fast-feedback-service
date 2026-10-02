# Place a built Python extension module into the source package.
#
# src/ffs is the single home for a built extension. An editable install
# makes it the only directory on the ffs package __path__, so a module
# installed anywhere else is unreachable, and setuptools package-data
# collects from there for a wheel. Both deployments therefore receive
# their extensions through pip, and nothing installs them separately.
#
# Usage:
#   include(FFSPythonModule)
#   ffs_python_module(<target> [<target>...])
#
# Each target must be a MODULE library, which is what nanobind_add_module
# creates.
#
# The placement runs on every build rather than only on a relink, so a
# module deleted from src/ffs by hand comes back. It cannot be expressed
# as a declared build output instead, because the file name is only known
# through a target-dependent generator expression and those are forbidden
# in a custom command's OUTPUT.
#
# The module is copied out of the build tree rather than built into
# src/ffs via LIBRARY_OUTPUT_DIRECTORY, because the 16- and 32-bit build
# directories would otherwise write the same path and each would treat
# the other's artifact as its own output already up to date.

cmake_minimum_required(VERSION 3.20)
include_guard()

set(FFS_PYTHON_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/src/ffs"
    CACHE PATH "In-tree Python package that built extension modules are placed in")

function(ffs_python_module)
    foreach(target IN LISTS ARGV)
        get_target_property(type ${target} TYPE)
        if(NOT type STREQUAL "MODULE_LIBRARY")
            message(FATAL_ERROR
                    "ffs_python_module: ${target} is a ${type}, expected a MODULE_LIBRARY")
        endif()
        add_custom_target(${target}_in_tree ALL
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:${target}>"
                    "${FFS_PYTHON_PACKAGE_DIR}/$<TARGET_FILE_NAME:${target}>"
            COMMENT "Placing ${target} in ${FFS_PYTHON_PACKAGE_DIR}"
            VERBATIM
        )
        add_dependencies(${target}_in_tree ${target})
    endforeach()
endfunction()
