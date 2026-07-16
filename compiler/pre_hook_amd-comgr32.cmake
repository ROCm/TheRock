# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-hook for 32-bit comgr build
# This file is loaded BEFORE the toolchain file, allowing us to override compiler settings

# 32-bit builds are only supported on Windows
if(NOT WIN32)
  message(FATAL_ERROR "32-bit builds (amd-comgr32) are only supported on Windows. "
                      "Attempted to build on ${CMAKE_SYSTEM_NAME}.")
endif()

# For Windows 32-bit builds, we need to use the x86 compiler
if(WIN32 AND NOT DEFINED CMAKE_C_COMPILER)
  # Get parent compiler path and replace x64 with x86
  string(REPLACE "/Hostx64/x64/" "/Hostx64/x86/" _c_compiler_x86 "${CMAKE_C_COMPILER}")
  string(REPLACE "/Hostx64/x64/" "/Hostx64/x86/" _cxx_compiler_x86 "${CMAKE_CXX_COMPILER}")

  # Set for this subproject
  set(CMAKE_C_COMPILER "${_c_compiler_x86}" CACHE FILEPATH "C compiler for x86" FORCE)
  set(CMAKE_CXX_COMPILER "${_cxx_compiler_x86}" CACHE FILEPATH "CXX compiler for x86" FORCE)

  # Set architecture
  set(CMAKE_SYSTEM_PROCESSOR "x86" CACHE STRING "Target processor" FORCE)
  set(CMAKE_SIZEOF_VOID_P 4 CACHE STRING "Size of void pointer" FORCE)

  message(STATUS "32-bit build: Using x86 compiler: ${CMAKE_C_COMPILER}")
endif()

# Enable ASAN for Comgr32 when THEROCK_SANITIZER is set to ASAN or HOST_ASAN
if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  set(ADDRESS_SANITIZER ON)
  message(STATUS "Enabling ASAN for Comgr32 (THEROCK_SANITIZER=${THEROCK_SANITIZER})")
endif()

if(THEROCK_BUILD_COMGR_TESTS)
  set(BUILD_TESTING ON CACHE BOOL "Enable comgr tests" FORCE)
else()
  set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)
endif()

# Use lib32 for 32-bit library installation
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;$ORIGIN/llvm/lib;$ORIGIN/rocm_sysdeps/lib")

# Set the DLL name to amd_comgr32.dll for 32-bit Windows builds
if(WIN32)
  set(COMGR_DLL_NAME "amd_comgr32.dll" CACHE STRING "Windows 32-bit DLL output name" FORCE)
  message(STATUS "32-bit comgr: Setting DLL name to ${COMGR_DLL_NAME}")
endif()
