# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT WIN32)
  # Configure roctracer if on a supported operating system (Linux).
  # rocBLAS has deprecated dependencies on roctracer. We apply a patch to redirect
  # naked linking against `-lroctx64` to an explicitly found version of the library.
  # See: https://github.com/ROCm/TheRock/issues/364
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
  find_library(_therock_legacy_roctx64 roctx64 REQUIRED)
  cmake_language(DEFER CALL therock_patch_linked_lib OLD_LIBRARY "roctx64" NEW_TARGET "${_therock_legacy_roctx64}")
endif()

# Enable BUILD_ADDRESS_SANITIZER when THEROCK_SANITIZER is ASAN
# This enables RCCL's LTO optimization bypass for faster ASAN link times
if(THEROCK_SANITIZER STREQUAL "ASAN")
  set(BUILD_ADDRESS_SANITIZER ON)
  message(STATUS "Enabling BUILD_ADDRESS_SANITIZER for RCCL
(THEROCK_SANITIZER=${THEROCK_SANITIZER})")

  # Work around a variable-shadowing issue: the sanitizer stanza in the
  # generated toolchain uses string(APPEND CMAKE_CXX_FLAGS ...) which creates
  # a normal variable that shadows the cache variable.  The cache variable
  # (populated from CMAKE_CXX_FLAGS_INIT) contains --hip-path and
  # --hip-device-lib-path, but the normal variable does not.  RCCL's
  # DeviceLinker.cmake reads CMAKE_CXX_FLAGS at configure time to extract
  # these flags, so we must ensure they are present in the normal variable.
  string(REGEX MATCHALL "--hip-path=[^ ]+" _hip_path "${CMAKE_CXX_FLAGS_INIT}")
  string(REGEX MATCHALL "--hip-device-lib-path=[^ ]+" _hip_devlib "${CMAKE_CXX_FLAGS_INIT}")
  foreach(_flag IN LISTS _hip_path _hip_devlib)
    if(NOT "${CMAKE_CXX_FLAGS}" MATCHES "${_flag}")
      string(APPEND CMAKE_CXX_FLAGS " ${_flag}")
    endif()
  endforeach()
endif()
