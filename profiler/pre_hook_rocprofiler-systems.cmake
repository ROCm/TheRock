# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_OFFLOAD_LIB_DIR)
  find_library(LIBOMPTARGET_SO NAMES omptarget
    HINTS "${THEROCK_OFFLOAD_LIB_DIR}")
endif()

find_package(OpenMP CONFIG)
if(TARGET OpenMP::omp AND NOT TARGET OpenMP::OpenMP_CXX)
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
  target_link_libraries(OpenMP::OpenMP_CXX INTERFACE OpenMP::omp)
endif()
if(TARGET OpenMP::omp AND NOT TARGET OpenMP::OpenMP_C)
  add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)
  target_link_libraries(OpenMP::OpenMP_C INTERFACE OpenMP::omp)
endif()

set(_rocprofsys_disable_examples "lulesh")
if(ROCM_BUILD_FORTRAN_LIBS)
  set(amdflang_EXECUTABLE "${CMAKE_Fortran_COMPILER}")
else()
  list(APPEND _rocprofsys_disable_examples "openmp-vv" "hpc")
endif()
set(ROCPROFSYS_DISABLE_EXAMPLES "${_rocprofsys_disable_examples}" CACHE STRING "" FORCE)
