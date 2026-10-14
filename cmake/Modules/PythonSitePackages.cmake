# Sets FFS_PYTHON_SITEARCH to the python extension module directory,
# relative to the install prefix. Include after find_package(Python).
#
# An absolute install DESTINATION overrides CMAKE_INSTALL_PREFIX, so a
# destination built from this lands inside the prefix the install is aimed
# at, unlike one built from ${Python_SITEARCH}.

include_guard(GLOBAL)

if(NOT Python_EXECUTABLE)
    message(FATAL_ERROR "find_package(Python) must run before including PythonSitePackages")
endif()

# platlib with both roots pinned to / yields the scheme's own layout with no
# prefix attached, which is precisely the relative form wanted here.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import os, sysconfig; print(os.path.relpath(sysconfig.get_path('platlib', vars={'base': '/', 'platbase': '/'}), '/'))"
    OUTPUT_VARIABLE _ffs_python_sitearch
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _ffs_python_sitearch_result
)

if(NOT _ffs_python_sitearch_result EQUAL 0 OR _ffs_python_sitearch MATCHES "^(/|\\.\\.)")
    message(FATAL_ERROR
        "Could not determine a prefix-relative site-packages path from "
        "${Python_EXECUTABLE} (got '${_ffs_python_sitearch}')")
endif()

set(FFS_PYTHON_SITEARCH "${_ffs_python_sitearch}" CACHE INTERNAL
    "Python site-packages directory, relative to CMAKE_INSTALL_PREFIX")

message(STATUS "Python modules will install to \${CMAKE_INSTALL_PREFIX}/${FFS_PYTHON_SITEARCH}")
