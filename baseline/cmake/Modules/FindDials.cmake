# Sets up a libtbx-Dials distribution
#
# There must be a defined variable DIALS_BUILD that points to the
# build directory to be used. Modules will be found using relative
# paths or other introspection.
#
# Once done this will define
#  DIALS_FOUND - System has Dials
#  DIALS_INCLUDE_DIRS - The Dials include directories
#  DIALS_LIBRARIES - The libraries needed to use Dials
#  DIALS_DEFINITIONS - Compiler switches required for using Dials
#
# And also sets up an imported Dials::Dials target, that allows basic
# usage of the dials infrastructure.



set(DIALS_BUILD ${DIALS_BUILD} CACHE PATH "Dials build folder")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Dials DEFAULT_MSG DIALS_BUILD)

if(DIALS_BUILD)
  if (NOT TARGET Dials::Dials)
    add_library(Dials::Dials IMPORTED INTERFACE)
  endif()

  set(DIALS_INCLUDE_DIRS "${DIALS_BUILD}/include;${DIALS_BUILD}/../modules;${DIALS_BUILD}/../modules/cctbx_project")
  mark_as_advanced(DIALS_INCLUDE_DIRS)

  set_target_properties(Dials::Dials PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${DIALS_INCLUDE_DIRS}"
    # INTERFACE_COMPILE_DEFINITIONS "${HDF5_DEFINITIONS_CLEAN}"
    INTERFACE_LINK_LIBRARIES      "Boost::boost"
    )
endif()