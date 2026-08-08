# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-hook for the 32-bit comgr build. Tests are disabled; only the shared
# library (amd_comgr32.dll) is produced.

set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)

if(WIN32)
  set(CMAKE_INSTALL_RPATH "")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/llvm/lib;$ORIGIN/rocm_sysdeps/lib")
endif()
