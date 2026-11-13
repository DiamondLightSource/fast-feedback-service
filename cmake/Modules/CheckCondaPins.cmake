# Warn when the conda environment the build uses has drifted from the pins.
#
# conda/development/environment.yml carries every version, so a pull that
# moves a pin leaves an existing environment behind until it is updated.
# The build then compiles against the old packages and fails in ways that
# do not name the cause, such as a link error between two fmt majors. This
# compares each pin against the environment and names every package that
# does not match.
#
# conda records each installed package as
# conda-meta/<name>-<version>-<build>.json, so a pin is met when a file with
# its name and version exists. conda reads name=1.2 as 1.2 or any 1.2.x, and
# a wildcard pin such as gcc=15.* becomes the same glob. The environment is
# the prefix holding the C++ compiler, since that is where the headers come
# from; a compiler from outside a conda environment has no conda-meta beside
# it and the check is skipped. On macOS the packages in conda/macos/packages
# are skipped as well, as derive.py drops them there.
#
# It only warns, since building against a different environment on purpose
# is allowed. Only the pinned packages are checked, not what they pull in.
# The pin file and the environment's history are configure dependencies, so
# a pull that moves a pin or an update to the environment reruns the check.

include_guard(GLOBAL)

# The environment is the prefix above the compiler's bin directory
cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _ffs_env_bin)
cmake_path(GET _ffs_env_bin PARENT_PATH _ffs_env)
if(NOT EXISTS "${_ffs_env}/conda-meta/history")
    return()
endif()

set(_ffs_pin_file "${CMAKE_SOURCE_DIR}/conda/development/environment.yml")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_ffs_pin_file}" "${_ffs_env}/conda-meta/history")
set(_ffs_update "mamba env update -p ${_ffs_env} -f ${CMAKE_SOURCE_DIR}/conda/development/environment.yml")

# macOS has no Linux compilers, so their pins do not apply there
set(_ffs_dropped "")
if(APPLE)
    set(_ffs_drop_file "${CMAKE_SOURCE_DIR}/conda/macos/packages")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_ffs_drop_file}")
    file(STRINGS "${_ffs_drop_file}" _ffs_lines)
    foreach(_ffs_line IN LISTS _ffs_lines)
        string(REGEX REPLACE "#.*" "" _ffs_line "${_ffs_line}")
        string(STRIP "${_ffs_line}" _ffs_line)
        if(_ffs_line)
            list(APPEND _ffs_dropped "${_ffs_line}")
        endif()
    endforeach()
    set(_ffs_update "python ${CMAKE_SOURCE_DIR}/conda/derive.py --drop ${CMAKE_SOURCE_DIR}/conda/macos/packages\n  mamba env update -p ${_ffs_env} -f ${CMAKE_SOURCE_DIR}/conda/macos/environment.yml")
endif()

# Each flat "  - name=version" entry, the only form derive.py accepts
file(STRINGS "${_ffs_pin_file}" _ffs_pins REGEX "^  - [A-Za-z0-9_.-]+=")
set(_ffs_meta "${_ffs_env}/conda-meta")
set(_ffs_stale "")
foreach(_ffs_pin IN LISTS _ffs_pins)
    string(REGEX REPLACE "#.*" "" _ffs_pin "${_ffs_pin}")
    string(REGEX MATCH "^  - ([A-Za-z0-9_.-]+)=([^ ]+)" _ffs_unused "${_ffs_pin}")
    set(_ffs_name "${CMAKE_MATCH_1}")
    set(_ffs_version "${CMAKE_MATCH_2}")
    if(_ffs_name IN_LIST _ffs_dropped)
        continue()
    endif()

    # The pinned version itself, or any release under it
    file(GLOB _ffs_found
        "${_ffs_meta}/${_ffs_name}-${_ffs_version}-*.json"
        "${_ffs_meta}/${_ffs_name}-${_ffs_version}.*-*.json")
    if(_ffs_found)
        continue()
    endif()

    # Name the version installed instead. A version starts with a digit,
    # which keeps python from matching python-dateutil.
    file(GLOB _ffs_installed RELATIVE "${_ffs_meta}" "${_ffs_meta}/${_ffs_name}-*.json")
    list(FILTER _ffs_installed INCLUDE REGEX "^${_ffs_name}-[0-9][^-]*-[^-]+\\.json$")
    if(_ffs_installed)
        list(GET _ffs_installed 0 _ffs_have)
        string(REGEX REPLACE "^${_ffs_name}-([^-]+)-.*" "\\1" _ffs_have "${_ffs_have}")
    else()
        set(_ffs_have "not installed")
    endif()
    list(APPEND _ffs_stale "${_ffs_name} ${_ffs_have}, pinned ${_ffs_version}")
endforeach()

if(_ffs_stale)
    list(JOIN _ffs_stale "\n  " _ffs_report)
    message(WARNING
        "The conda environment at ${_ffs_env} does not match the pins in "
        "conda/development/environment.yml:\n  ${_ffs_report}\n"
        "Update it with:\n  ${_ffs_update}")
endif()
