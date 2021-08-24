#[[.md:

# FindFPGA

]]

include(CheckCXXCompilerFlag)
include (FindPackageHandleStandardArgs)

check_cxx_compiler_flag(-fintelfpga CXX_HAS_FPGA_FLAG)

if(CXX_HAS_FPGA_FLAG)
    set(CXX_HAS_FPGA_FLAG Yes)
    # FPGA board selection
    if(NOT DEFINED FPGA_BOARD)
        # Boards are defined as package:board
        # Packages from: "aoc -list-boards" and "aoc -list-board-packages"
        set(FPGA_BOARD "intel_s10sx_pac:pac_s10" CACHE STRING "The package:board combination to pass to aoc")
        message(STATUS "FPGA_BOARD not specified, using default")
    endif()
    message(STATUS "Configuring for FPGA board: ${FPGA_BOARD}")


    add_library(FPGA::EMULATED INTERFACE IMPORTED )
    set_target_properties(FPGA::EMULATED PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "FPGA_EMULATOR;FPGA"
        INTERFACE_COMPILE_OPTIONS "-fintelfpga"
        INTERFACE_LINK_OPTIONS "-fintelfpga")

    add_library(FPGA::FPGA INTERFACE IMPORTED)
    set_target_properties(FPGA::FPGA PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "FPGA"
        INTERFACE_COMPILE_OPTIONS "-fintelfpga"
        INTERFACE_LINK_OPTIONS "-fintelfpga;-Xshardware;-Xsboard=${FPGA_BOARD}")
endif()

find_package_handle_standard_args(FPGA REQUIRED_VARS FPGA_BOARD CXX_HAS_FPGA_FLAG)