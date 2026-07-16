# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-hook for 32-bit OpenCL runtime build
# 32-bit builds are only supported on Windows

# 32-bit builds are only supported on Windows
if(NOT WIN32)
  message(FATAL_ERROR "32-bit builds (core-ocl32) are only supported on Windows. "
                      "Attempted to build on ${CMAKE_SYSTEM_NAME}.")
endif()

# When ocl-clr32 does find_package(amd_comgr), the dependency provider will look
# for the "amd_comgr" package. Since we provide "amd_comgr32" to avoid conflicts,
# we need to add "amd_comgr" as an alias in the provided packages list.
# This runs BEFORE the dependency provider is loaded in the init file.
if(DEFINED THEROCK_PROVIDED_PACKAGES)
  list(APPEND THEROCK_PROVIDED_PACKAGES "amd_comgr")
  # Point amd_comgr lookups to the amd-comgr32 installation (note the /lib in the path)
  # Use THEROCK_BINARY_DIR which is the super-project binary dir, not CMAKE_BINARY_DIR
  set(THEROCK_PACKAGE_DIR_amd_comgr "${THEROCK_BINARY_DIR}/compiler/amd-comgr32/stage/lib32/lib/cmake/amd_comgr" CACHE PATH "" FORCE)
endif()

# Use lib32 for 32-bit library installation
if(NOT WIN32)
  set(AMDOCL_INSTALL_LIBDIR "opencl32")
  set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../lib32")
else()
  # Windows uses standard DLL search path
  set(AMDOCL_INSTALL_LIBDIR "opencl32")
endif()

