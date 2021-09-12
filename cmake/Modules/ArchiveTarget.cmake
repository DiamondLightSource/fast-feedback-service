#[[.md:

# ArchiveTarget.cmake

Utility function for saving a commit ID on build, then archiving the
output target once it has been built. Designed to make it slightly
harder to accidentally delete fpga-build targets, which generally have a
minumum compile time of ~2 hours.

Usage:

    include(ArchiveTarget)
    archive_target(<target>)

        Adds pre and post-build stages to the target <target>. These
        will save the commit ID as reported by "git describe --tags
        --always --dirty", and then copy the resultant target to
        <archive_dir>/<target_name>.<commit_id> on completion.

        The <archive_dir> is controlled by the TARGET_ARCHIVE_DIR
        variable, which defaults to `${CMAKE_BINARY_DIR}/archive/`.

Usage as a Script:

    cmake -P ArchiveTarget.cmake -- (get_commit <output_name> | archive <commit_id_file> <target>)

Commands:
    get_commit <output_name>

        Read the commit ID from git, using "git describe", with the
        flags `--tags --always --dirty`. The result will be written to
        the file <output_name>. This should be called as part of a
        pre-build custom step so that the commit ID is saved at the
        start of compilation (so that the archived name actually
        represents the status at start of compilation).
    
    archive <commit_id_file> <target_file> [<archive_dir>]

        Archive a target binary <target_file> to <archive_dir>, or
        `archive/` if unspecified. This will be copied to the file:
        
            `<archive_dir>/<target_file basename>.<commit_id>`

        If the file already exists, then both source and destination
        will be compared; if they are identical, nothing is done.
        Otherwise, the target will be archived to `<basename>.<commit_id>.<int>`,
        where `<int>` is the smallest integer that doesn't already exist.
]]


cmake_minimum_required(VERSION 3.20)
include_guard()

set(TARGET_ARCHIVE_DIR "${CMAKE_BINARY_DIR}/archive" CACHE PATH "Output path for cached copied of output targets")

# Check to see if we are being run with include(). In this case, we want
# to provide macros to run this same file in script mode
if(NOT CMAKE_SCRIPT_MODE_FILE OR CMAKE_PARENT_LIST_FILE)
    # Archives a target on build completion with the commit ID at the point
    # compilation was started
    function(archive_target target)
        add_custom_command(TARGET ${target} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
                    -- get_commit "${CMAKE_BINARY_DIR}/commit.id")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
                    -- archive "${CMAKE_BINARY_DIR}/commit.id" $<TARGET_FILE:${target}> "${TARGET_ARCHIVE_DIR}")
    endfunction()
    # Don't do any of the actual script-file parts
    return()
endif()

########################################################################
# ArchiveTarget.cmake script file parts

set(USAGE "cmake -P Archive.cmake -- (get_commit <output_name> | archive <commit_id_file> <target> [<archive_dir>])" )

# Do some general argument processing;
# find items after "--" and read them into an ARGS variable
foreach(_arg RANGE ${CMAKE_ARGC})
    math(EXPR _next_arg "${_arg} + 1")
    if ("${CMAKE_ARGV${_arg}}" STREQUAL "--")
        break()
    endif()
endforeach()
# Copy this into a separate argument list
foreach(_arg RANGE ${_next_arg} ${CMAKE_ARGC})
    if(CMAKE_ARGV${_arg})
        list(APPEND ARGS "${CMAKE_ARGV${_arg}}")
    endif()
endforeach()

list(LENGTH ARGS _arg_len)
list(GET ARGS 0 _command)
if (_command STREQUAL "get_commit")
    # We want to read a description of the current commit into a file
    if (NOT _arg_len EQUAL 2)
        message(FATAL_ERROR "Unexpected command arguments for get_commit.\nUsage: ${USAGE}")
    endif()
    list(GET ARGS 1 outfile)
    execute_process(COMMAND git describe --dirty --always --tags
                    ERROR_QUIET
                    RESULT_VARIABLE git_describe_results
                    OUTPUT_VARIABLE git_describe_output
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(git_describe_results)
        file(WRITE "${outfile}" "")
        message(WARNING "Archiver: Could not run \"git describe\", target will not be archived.")
        return()
    endif()
    file(WRITE "${outfile}" "${git_describe_output}")
    message(STATUS "Archiver: Got git commit id ${git_describe_output}")
elseif(_command STREQUAL "archive")
    if (NOT (_arg_len EQUAL 3 OR _arg_len EQUAL 4))
        message(FATAL_ERROR "Unexpected command arguments for get_commit.\nUsage: ${USAGE}")
    endif()
    list(GET ARGS 1 commit_id_file)
    list(GET ARGS 2 archive_target)
    # Get the archive dir, if specified
    if (_arg_len EQUAL 4)
        list(GET ARGS 3 archive_path)
    else()
        # Get the archive dir as the same as the target plus archive/
        cmake_path(GET archive_target PARENT_PATH archive_path)
        cmake_path(APPEND archive_path "archive")
    endif()

    if (NOT EXISTS "${archive_target}")
        message(FATAL_ERROR "Target file does not exist: ${archive_target}")
    elseif(NOT EXISTS "${commit_id_file}")
        message(FATAL_ERROR "Commit ID source file does not exist: ${commit_id_file}")
    endif()
    file(READ "${commit_id_file}" commit_id)
    if (NOT commit_id)
        # If this file is empty, then we couldn't get a commit ID. We
        # should have already been warned about this.
        return()
    endif()
    # Work out the target name for archiving
    cmake_path(GET archive_target FILENAME _target_basename)
    cmake_path(APPEND archive_path "${_target_basename}.${commit_id}" )
    # Make sure the output path exists
    file(MAKE_DIRECTORY "${_archive_path}")
    # If the file exists, then we might be rebuilding.
    # Check for duplication, and rename if not the same
    if(EXISTS "${archive_path}")
        file(SHA256 "${archive_path}" _dest_hash)
        file(SHA256 "${archive_target}" _target_hash)
        if(_dest_hash STREQUAL _target_hash)
            message(STATUS "Archiver: Nothing to do for ${_target_basename}")
            return()
        else()
            # We have a conflicting file. Find an integer to append that
            # doesn't already exist
            set(copy_integer 1)
            while(EXISTS "${archive_path}.${copy_integer}")
                math(EXPR copy_integer "${copy_integer} + 1")
                if(copy_integer GREATER 100)
                    message(WARNING "Archiver: Gave up trying to find unused name for target ${archive_path}")
                    return()
                endif()
            endwhile()
            set(archive_path "${archive_path}.${copy_integer}")
        endif()
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${archive_target}" "${archive_path}"
    RESULT_VARIABLE _copy_result)
    if(_copy_result)
        message(WARNING "Archiver: Error copying ${archive_target} to ${archive_path}, not archived")
    endif()
    cmake_path(GET archive_path FILENAME _archive_filename)
    message(STATUS "Archived ${_target_basename} to archive/${_archive_filename}")
else()
    if(NOT _command)
        message(FATAL_ERROR "No command.\nUsage: ${USAGE}")
    else()
        message(FATAL_ERROR "Unknown command: ${_command}.\nUsage: ${USAGE}")
    endif()
endif()
