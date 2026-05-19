# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This finder resolves the virtual BLAS package for sub-projects.
# It defers to the built host-blas, if available, otherwise, failing.
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

if(("OpenBLAS" IN_LIST THEROCK_PROVIDED_PACKAGES) OR ("OpenBLAS64" IN_LIST
                                                     THEROCK_PROVIDED_PACKAGES))
  message(STATUS "Resolving bundled host-blas library from super-project")

  set(_want_ilp64 FALSE)
  if(DEFINED BLA_SIZEOF_INTEGER AND BLA_SIZEOF_INTEGER EQUAL 8)
    set(_want_ilp64 TRUE)
  endif()

  if(_want_ilp64)
    if(NOT "OpenBLAS64" IN_LIST THEROCK_PROVIDED_PACKAGES)
      message(
        FATAL_ERROR
          "BLA_SIZEOF_INTEGER=8 (ILP64) requested but the super-project did not "
          "provide OpenBLAS64. Add therock-host-blas64 to this subproject's "
          "BUILD_DEPS or RUNTIME_DEPS.")
    endif()
    find_package(OpenBLAS64 CONFIG REQUIRED)
    set(_OPENBLAS OpenBLAS64)
  else()
    if(NOT "OpenBLAS" IN_LIST THEROCK_PROVIDED_PACKAGES)
      message(
        FATAL_ERROR
          "LP64 LAPACK requested but the super-project did not provide OpenBLAS. "
          "Add therock-host-blas, or set BLA_SIZEOF_INTEGER=8 if only OpenBLAS64 "
          "is available.")
    endif()
    find_package(OpenBLAS CONFIG REQUIRED)
    set(_OPENBLAS OpenBLAS)
  endif()

  # See: https://cmake.org/cmake/help/latest/module/FindLAPACK.html
  set(LAPACK_LINKER_FLAGS)
  set(LAPACK_LIBRARIES ${_OPENBLAS}::OpenBLAS)
  add_library(LAPACK::LAPACK ALIAS ${_OPENBLAS}::OpenBLAS)
  set(LAPACK95_LIBRARIES)
  set(LAPACK95_FOUND FALSE)
  set(LAPACK_FOUND TRUE)
  cmake_policy(POP)
else()
  cmake_policy(POP)
  set(LAPACK_FOUND FALSE)
endif()
