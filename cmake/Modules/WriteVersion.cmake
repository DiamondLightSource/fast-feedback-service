# Resolve the version and write it into the generated sources.
#
# Runs two ways: included by ResolveVersion at configure time, and as a
# `cmake -P` script on every build of a plain cmake tree. Both read the same
# inputs:
#
#   FFS_SOURCE_DIR, FFS_BINARY_DIR
#       The source tree to resolve and the build tree to write into.
#   FFS_VERSION_FULL
#       A version already resolved by the python build. When empty, it is
#       resolved here with setuptools_scm.
#   Python_EXECUTABLE
#       The interpreter to run setuptools_scm with.
#
# setuptools_scm is run from the command line rather than reimplemented, so
# it reads the same pyproject.toml configuration as the python build and
# honours SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FFS for a tree without git.
#
# configure_file only rewrites an output whose content changed, so an
# unchanged version leaves version.cc untouched and nothing relinks.

if(NOT FFS_VERSION_FULL)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m setuptools_scm --root "${FFS_SOURCE_DIR}"
        WORKING_DIRECTORY "${FFS_SOURCE_DIR}"
        RESULT_VARIABLE _ffs_scm_result
        OUTPUT_VARIABLE FFS_VERSION_FULL
        ERROR_VARIABLE _ffs_scm_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _ffs_scm_result EQUAL 0 OR NOT FFS_VERSION_FULL)
        set(FFS_VERSION_FULL "0.0.0.dev0+unknown")
        message(WARNING
            "Could not resolve the version with setuptools_scm under "
            "${Python_EXECUTABLE}; using ${FFS_VERSION_FULL}. Install "
            "setuptools_scm into that environment, or set "
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FFS for a tree without git.\n"
            "${_ffs_scm_error}")
    endif()
endif()

configure_file("${FFS_SOURCE_DIR}/version.cc.in" "${FFS_BINARY_DIR}/version.cc" @ONLY)
configure_file("${FFS_SOURCE_DIR}/src/ffs/_version.py.in" "${FFS_BINARY_DIR}/_version.py" @ONLY)
