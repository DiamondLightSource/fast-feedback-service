# Resolve CUDA_ARCH into CMAKE_CUDA_ARCHITECTURES.
#
# CUDA_ARCH accepts:
#   native          the compute capability of the GPU in this machine
#                   (the default), so a local build produces one cubin
#   all-supported   every architecture the published image ships, for
#                   deployment builds
#   anything else   a bare compute capability number, a ";"-separated
#                   list of them, or one of the labelled entries offered
#                   in the cache selector below
#
# Exports FFS_CUDA_ARCH_REAL, FFS_CUDA_ARCH_VIRTUAL and
# FFS_DETECTED_GPU_ARCH as bare compute capability numbers, which the
# root script uses to set the build.
#
# Must be included before project(), because CMAKE_CUDA_ARCHITECTURES is
# read when a subdirectory enables the CUDA language.

# Only do this if we're being called from the root CMakeLists
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Architectures the published container image is built for: one cubin
  # per compute capability target, so the image needs no JIT anywhere
  # from Turing to Blackwell.
  set(_ffs_deploy_real 75 80 86 89 90 100 103 120 121)

  # PTX to fall back on when no cubin above matches, so the driver JITs
  # instead of refusing to load. compute_120 catches cards newer than
  # anything we compiled for; compute_103 is the only route onto Thor
  # (sm_110), since cubins never cross a major revision but PTX does.
  set(_ffs_deploy_virtual 103 120)

  # Compute capability to fall back on when there is no GPU to probe.
  # Turing is the oldest CUDA 13 still compiles for; it dropped Maxwell,
  # Pascal and Volta.
  set(_ffs_fallback_arch 75)

  set(CUDA_ARCH "native" CACHE STRING "CUDA compute capability target architecture")
  set_property(CACHE CUDA_ARCH PROPERTY STRINGS
      "native"
      "all-supported"
      "75 - Turing (RTX 20xx, Quadro RTX)"
      "80 - Ampere (A100)"
      "86 - Ampere (RTX 30xx, A40)"
      "89 - Ada Lovelace (RTX 40xx, L40S)"
      "90 - Hopper (H100, GH200)"
      "100 - Blackwell (B200, GB200)"
      "103 - Blackwell (B300, GB300)"
      "120 - Blackwell (RTX 50xx, RTX PRO)"
      "121 - Blackwell (DGX Spark GB10)"
  )

  # Probe the local GPU once. This decides the target for "native" and
  # is advisory otherwise, so it runs unconditionally and sets
  # FFS_DETECTED_GPU_ARCH to a bare compute capability number or empty
  # if no GPU is present.
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _ffs_detected_caps
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  # Take the first GPU and convert the "X.Y" string to a bare number
  # "XY" for comparison with the requested architectures.
  string(REGEX MATCH "[0-9]+\\.[0-9]+" _ffs_first_cap "${_ffs_detected_caps}")
  string(REPLACE "." "" FFS_DETECTED_GPU_ARCH "${_ffs_first_cap}")

  if(FFS_DETECTED_GPU_ARCH)
    message(STATUS "Detected GPU compute capability: sm_${FFS_DETECTED_GPU_ARCH}")
  endif()

  if(CUDA_ARCH STREQUAL "native")
    if(FFS_DETECTED_GPU_ARCH)
      set(FFS_CUDA_ARCH_REAL ${FFS_DETECTED_GPU_ARCH})
    else()
      message(STATUS
        "No GPU detected for CUDA_ARCH=native; falling back to sm_${_ffs_fallback_arch}. "
        "Pass -DCUDA_ARCH=<capability> to target a specific card, or "
        "-DCUDA_ARCH=all-supported to build what the container ships."
      )
      set(FFS_CUDA_ARCH_REAL ${_ffs_fallback_arch})
    endif()
    # No PTX: a native build targets exactly the card in front of you, so
    # a JIT fallback would only inflate the binary.
    set(FFS_CUDA_ARCH_VIRTUAL "")
  elseif(CUDA_ARCH STREQUAL "all-supported")
    set(FFS_CUDA_ARCH_REAL ${_ffs_deploy_real})
    set(FFS_CUDA_ARCH_VIRTUAL ${_ffs_deploy_virtual})
  else()
    set(FFS_CUDA_ARCH_REAL "")
    foreach(_entry IN LISTS CUDA_ARCH)
      # Pull the leading number off each entry, so "80;90" becomes "80"
      # and "90", and "86 - Ampere" becomes "86".
      string(REGEX MATCH "^[0-9]+" _number "${_entry}")
      if(NOT _number)
        message(FATAL_ERROR
          "CUDA_ARCH entry \"${_entry}\" does not begin with a compute "
          "capability number. Expected something like 86, \"80;90\", "
          "native, or all-supported."
        )
      endif()
      list(APPEND FFS_CUDA_ARCH_REAL ${_number})
    endforeach()
    list(REMOVE_DUPLICATES FFS_CUDA_ARCH_REAL)
    list(SORT FFS_CUDA_ARCH_REAL COMPARE NATURAL)
    # PTX for the newest capability asked for, so a card newer than
    # anything we compiled JITs instead of failing to load.
    list(GET FFS_CUDA_ARCH_REAL -1 FFS_CUDA_ARCH_VIRTUAL)
  endif()

  set(CMAKE_CUDA_ARCHITECTURES "")
  foreach(_arch IN LISTS FFS_CUDA_ARCH_REAL)
    list(APPEND CMAKE_CUDA_ARCHITECTURES "${_arch}-real")
  endforeach()
  foreach(_arch IN LISTS FFS_CUDA_ARCH_VIRTUAL)
    list(APPEND CMAKE_CUDA_ARCHITECTURES "${_arch}-virtual")
  endforeach()

  message(STATUS "CUDA architectures (CUDA_ARCH=${CUDA_ARCH}): ${CMAKE_CUDA_ARCHITECTURES}")
endif()
