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
endif()

# RCCL later force-sets CMAKE_C/CXX_FLAGS_RELEASE to -O3, which removes the
# compact host -Oz. Restoring it is deferred to the end of this directory:
# compile rules read those cache entries at generate time.
if(NOT COMMAND therock_asan_compact_preserve_host_oz)
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
endif()
therock_asan_compact_preserve_host_oz()
