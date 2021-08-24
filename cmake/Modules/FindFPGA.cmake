#[[.md:

# FindFPGA

]]

include(CheckCXXCompilerFlag)
include (FindPackageHandleStandardArgs)

check_cxx_compiler_flag(-fintelfpga CXX_HAS_FPGA_FLAG)

## Duplicate a target and all relevant properties
##
## This is an INTERNAL function to FindFPGA. It doesn't attempt to
## perfectly copy all properties, as WIP development is ongoing the
## more important properties might be expanded.
function(_duplicate_target new_target target)
    get_target_property(_type ${target} TYPE)
    if(_type STREQUAL STATIC_LIBRARY)
        add_library(${new_target} STATIC )
    elseif(_type STREQUAL MODULE_LIBRARY)
        add_library(${new_target} MODULE )
    elseif(_type STREQUAL SHARED_LIBRARY)
        add_library(${new_target} SHARED )
    elseif(_type STREQUAL EXECUTABLE)
        add_executable(${new_target} )
    endif()


    set(copy_properties "")
    # Build list of namespaced properties
    foreach(namespace "" INTERFACE)
        set(_prefix "")
        if(namespace)
            set(_prefix "${namespace}_")
        endif()
        foreach(_prop
            COMPILE_DEFINITIONS COMPILE_FEATURES COMPILE_OPTIONS INCLUDE_DIRECTORIES
            LINK_DEPENDS LINK_DIRECTORIES LINK_LIBRARIES LINK_OPTIONS
            POSITION_INDEPENDENT_CODE PRECOMPILE_HEADERS SOURCES SYSTEM_INCLUDE_DIRECTORIES
            )
            set(_pro "${_prefix}${_prop}")
            list(APPEND copy_properties "${_prop}")
        endforeach()
    endforeach()
    # "Other" properties
    # list(APPEND copy_properties LOCATION)

    # Now, copy all the propertes to the new target
    foreach(_property ${copy_properties})
        get_target_property(_prop ${target} ${_property})
        # Copy anything as long as it is not-NOTFOUND
        if(NOT _prop MATCHES ".*-NOTFOUND$")
            set_target_properties(${new_target} PROPERTIES ${_property} "${_prop}")
        endif()
    endforeach()
endfunction()

if(CXX_HAS_FPGA_FLAG)
    # This is a Windows-specific flag that enables exception handling in host code
    if(WIN32)
        set(FPGA_WIN_FLAG "/EHsc")
    endif()

    set(CXX_HAS_FPGA_FLAG Yes)
    # FPGA board selection
    if(NOT DEFINED FPGA_BOARD)
        # Boards are defined as package:board
        # Packages from: "aoc -list-boards" and "aoc -list-board-packages"
        set(FPGA_BOARD "intel_s10sx_pac:pac_s10" CACHE STRING "The package:board combination to pass to aoc")
        message(STATUS "FPGA_BOARD not specified, using default")
    endif()
    message(STATUS "Configuring for FPGA board: ${FPGA_BOARD}")

    add_custom_target(fpga)
    set_target_properties(fpga PROPERTIES EXCLUDE_FROM_ALL yes)

    add_custom_target(fpga_report)
    set_target_properties(fpga_report PROPERTIES EXCLUDE_FROM_ALL yes)

    add_library(FPGA::EMULATED INTERFACE IMPORTED )
    set_target_properties(FPGA::EMULATED PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "FPGA_EMULATOR;FPGA"
        INTERFACE_COMPILE_OPTIONS "-fintelfpga;${FPGA_WIN_FLAG}"
        INTERFACE_LINK_OPTIONS "-fintelfpga")

    add_library(FPGA::FPGA INTERFACE IMPORTED)
    set_target_properties(FPGA::FPGA PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "FPGA"
        INTERFACE_COMPILE_OPTIONS "-fintelfpga;${FPGA_WIN_FLAG}"
        INTERFACE_LINK_OPTIONS "-fintelfpga;-Xshardware;-Xsboard=${FPGA_BOARD}")
    
    ## Add FPGA variants of a target
    ##
    ## This will duplicate the target, and add variants for:
    ## - target.fpga:       Hardware FPGA (Adds -DFPGA compile definition)
    ## - target.fpga_emu:   Emulated FPGA (Adds -DFPGA and -DFPGA_EMULATOR compile definition)
    ## - target_report.a:   FPGA Report (Hardware FPGA, but only the early linking stages and report output)
    function(fpga_add_variants target)
        get_target_property(_imported ${target} IMPORTED)
        if(_imported)
            # Our duplicate target function doesn't handle this yet
            message(SEND_ERROR "Cannot add FPGA to an imported target")
        endif()

        # FPGA Emulator
        _duplicate_target(${target}.fpga_emu ${target})
        target_link_libraries(${target}.fpga_emu FPGA::EMULATED)
        
        # FPGA Report
        _duplicate_target(${target}_report.a ${target})
        target_link_libraries(${target}_report.a FPGA::FPGA)
        target_link_options(${target}_report.a PRIVATE "-fsycl-link=early")
        set_target_properties(${target}_report.a PROPERTIES EXCLUDE_FROM_ALL yes)
        add_dependencies(fpga_report ${target}_report.a)
    
        # FPGA hardware build
        _duplicate_target(${target}.fpga ${target})
        target_link_libraries(${target}.fpga FPGA::FPGA)
        set_target_properties(${target}.fpga PROPERTIES EXCLUDE_FROM_ALL yes)
        add_dependencies(fpga ${target}.fpga)
    endfunction()

    ## Convenience function to add a target and variants at the same time
    function(fpga_add_executable name)
        cmake_parse_arguments(PARSE_ARGV 1 _addexec "" "" "")
        add_executable(${name}.fpga_emu ${_addexec_UNPARSED_ARGUMENTS})
        target_link_libraries(${name}.fpga_emu FPGA::EMULATED)

        add_executable(${name}_report.a ${_addexec_UNPARSED_ARGUMENTS})
        target_link_libraries(${name}_report.a FPGA::FPGA)
        target_link_options(${name}_report.a PRIVATE "-fsycl-link=early")
        set_target_properties(${name}_report.a PROPERTIES EXCLUDE_FROM_ALL yes)
        add_dependencies(fpga_report ${name}_report.a)

        add_executable(${name}.fpga ${_addexec_UNPARSED_ARGUMENTS})
        target_link_libraries(${name}.fpga FPGA::FPGA)
        set_target_properties(${name}.fpga PROPERTIES EXCLUDE_FROM_ALL yes)
        add_dependencies(fpga ${name}.fpga)
    endfunction()
endif()

find_package_handle_standard_args(FPGA REQUIRED_VARS FPGA_BOARD CXX_HAS_FPGA_FLAG)