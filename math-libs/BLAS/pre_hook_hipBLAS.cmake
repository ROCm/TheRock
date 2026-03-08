# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The OpenMP config-mode cmake file (openmp-config.cmake) adds the clang
# resource dir as -isystem via INTERFACE_INCLUDE_DIRECTORIES on OpenMP::omp,
# which disrupts header search order when compiling HIP device code (clang's
# stdint.h gets found before the system one, breaking GCC's <cstdint>).
#
# Fix: pre-create the OpenMP imported targets so that openmpTargets.cmake's
# guard (checking if OpenMP::omp already exists) causes it to return early,
# skipping the problematic INTERFACE_INCLUDE_DIRECTORIES.
find_library(_omp_lib omp PATHS "${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/lib" NO_DEFAULT_PATH)
if(NOT TARGET OpenMP::omp)
  add_library(OpenMP::omp SHARED IMPORTED)
  if(_omp_lib)
    set_target_properties(OpenMP::omp PROPERTIES IMPORTED_LOCATION "${_omp_lib}")
  endif()
endif()
if(NOT TARGET OpenMP::OpenMP_CXX)
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
  set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-fopenmp"
    INTERFACE_LINK_OPTIONS "-fopenmp"
  )
  if(_omp_lib)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES INTERFACE_LINK_LIBRARIES "${_omp_lib}")
  endif()
endif()
if(NOT TARGET OpenMP::OpenMP_C)
  add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)
  set_target_properties(OpenMP::OpenMP_C PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-fopenmp"
    INTERFACE_LINK_OPTIONS "-fopenmp"
  )
  if(_omp_lib)
    set_target_properties(OpenMP::OpenMP_C PROPERTIES INTERFACE_LINK_LIBRARIES "${_omp_lib}")
  endif()
endif()
set(OpenMP_FOUND TRUE CACHE BOOL "" FORCE)
set(OpenMP_CXX_FOUND TRUE CACHE BOOL "" FORCE)
set(OpenMP_C_FOUND TRUE CACHE BOOL "" FORCE)
set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "" FORCE)
set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "" FORCE)
