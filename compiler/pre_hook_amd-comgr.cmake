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

# Debug info for comgr's own objects only; statically linked LLVM is left out
# to keep the PDB small.
# /Z7 rather than /Zi to avoid collision with LLVM's shared PCH (error C2859).
# /OPT:REF,/OPT:ICF restore the Release defaults that /DEBUG turns off.
# CMAKE_HOST_WIN32 rather than WIN32/MSVC, which are unset before project().
if(CMAKE_HOST_WIN32 AND THEROCK_FLAG_WINDOWS_DRIVER_BUILD)
  string(APPEND CMAKE_C_FLAGS " /Z7")
  string(APPEND CMAKE_CXX_FLAGS " /Z7")
  add_link_options("LINKER:/DEBUG,/OPT:REF,/OPT:ICF")
endif()
