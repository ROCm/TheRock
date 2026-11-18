# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Enable ASAN for Comgr when THEROCK_SANITIZER is set to ASAN or HOST_ASAN
if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  set(ADDRESS_SANITIZER ON)
  message(STATUS "Enabling ASAN for Comgr (THEROCK_SANITIZER=${THEROCK_SANITIZER})")
endif()

if(THEROCK_BUILD_COMGR_TESTS)
  set(BUILD_TESTING ON CACHE BOOL "Enable comgr tests" FORCE)
else()
  set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)
endif()

set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/llvm/lib;$ORIGIN/rocm_sysdeps/lib")

# Debug info for comgr's own objects only, not the statically linked LLVM.
# Embedded (/Z7): /Zi collides with LLVM's shared PCH. Needs CMP0141 (CMakeLists.txt).
# /OPT:REF,/OPT:ICF restore the Release defaults that /DEBUG turns off.
# CMAKE_HOST_WIN32: WIN32/MSVC are unset before project().
if(CMAKE_HOST_WIN32 AND THEROCK_FLAG_WINDOWS_DRIVER_BUILD)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
  add_link_options("LINKER:/DEBUG,/OPT:REF,/OPT:ICF")
endif()
