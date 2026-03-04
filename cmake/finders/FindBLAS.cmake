# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This finder resolves the virtual BLAS package for sub-projects.
# It defers to the built host-blas, if available, otherwise, failing.
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

if("OpenBLAS" IN_LIST THEROCK_PROVIDED_PACKAGES)
  cmake_policy(POP)
  message(STATUS "Resolving bundled host-blas library from super-project")
  find_package(OpenBLAS CONFIG REQUIRED)
  # See: https://cmake.org/cmake/help/latest/module/FindBLAS.html
  set(BLAS_LINKER_FLAGS)
  set(BLAS_LIBRARIES OpenBLAS::OpenBLAS)
  add_library(BLAS::BLAS ALIAS OpenBLAS::OpenBLAS)
  set(BLAS95_LIBRARIES)
  set(BLAS95_FOUND FALSE)
  set(BLAS_FOUND TRUE)
else()
  cmake_policy(POP)
  set(BLAS_FOUND FALSE)
endif()
