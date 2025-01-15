# Resolve the version by querying git and turn into a PEP-440 number
#
# Exports:
#   FFS_VERSION
#       Either X.Y.Z or X.Y.0.devN
#   FFS_VERSION_FULL
#           Either X.Y.Z+gSHA or X.Y.0.devZ+gSHA depending on 
#   FFS_VERSION_CMAKE
#       A limited-detail expression of the version number, suitable for
#       the internal CMake version handling (which knows nothing about
#       things like dev, rc, prerelease). Dev versions will have the
#       X.Y.0 of the next release, and non-dev branches will be plain
#       X.Y.Z.

find_package(Git QUIET)

if(NOT Git_FOUND)
    set(FFS_VERSION_FULL "0.0.0.dev0+g000000")
    set(FFS_VERSION_CMAKE "0.0.0")
    message(WARNING "No git, could not determine repository version")
else()
    execute_process(
        COMMAND
        "${GIT_EXECUTABLE}" describe --tags --long --first-parent
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE res
        OUTPUT_VARIABLE REPO_VERSION
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT REPO_VERSION)
        set(FFS_VERSION_FULL "0.0.0.dev0+g000000")
        set(FFS_VERSION_CMAKE "0.0.0")
        message(WARNING "Error determining version from git, ignoring.")
    else()
        # # Uncomment to recalculate version number on every commit... this might
        # # be annoying, so "live" with the lagging number on developer installs,
        # # but in places where we care (e.g. compiling for deployment) this will work
        # set_property(GLOBAL APPEND
        #     PROPERTY CMAKE_CONFIGURE_DEPENDS
        #     "${CMAKE_SOURCE_DIR}/.git/index"
        # )

        # Extract parts from the version number
        string(REGEX REPLACE "^v([0-9]+)\\.([0-9]+)(.*)-([0-9]+)-(g.+)$" "\\1;\\2;\\3;\\4;\\5" _ver_parts "${REPO_VERSION}")
        list(GET _ver_parts 0 _ver_version_major)
        list(GET _ver_parts 1 _ver_version_minor)
        list(GET _ver_parts 2 _ver_version_dev)
        list(GET _ver_parts 3 _ver_commits)
        list(GET _ver_parts 4 _ver_shasum)
        message(STATUS "${_ver_version_dev}")
        if (_ver_version_dev STREQUAL ".dev")
            # A dev build
            set(FFS_VERSION_FULL "${_ver_version_major}.${_ver_version_minor}.0.dev${_ver_commits}+${_ver_shasum}")
            set(FFS_VERSION_CMAKE "${_ver_version_major}.${_ver_version_minor}.0")
        else()
            # Is not a dev build, so a release or release branch
            set(FFS_VERSION_FULL "${_ver_version_major}.${_ver_version_minor}.${_ver_commits}+${_ver_shasum}")
            set(FFS_VERSION_CMAKE "${_ver_version_major}.${_ver_version_minor}.${_ver_commits}")
        endif()
    endif()
    
endif()