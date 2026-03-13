# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# rocprofiler-systems bundles Dyninst, which links against the legacy
# FindOpenMP target names (OpenMP::OpenMP_CXX, etc.).  The LLVM OpenMP
# config-file package only creates OpenMP::omp, so we shim the legacy
# names after the provider resolves the find.

find_package(OpenMP CONFIG)
if(TARGET OpenMP::omp AND NOT TARGET OpenMP::OpenMP_CXX)
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
  target_link_libraries(OpenMP::OpenMP_CXX INTERFACE OpenMP::omp)
endif()
if(TARGET OpenMP::omp AND NOT TARGET OpenMP::OpenMP_C)
  add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)
  target_link_libraries(OpenMP::OpenMP_C INTERFACE OpenMP::omp)
endif()

# Some examples manually compile fortran and assume flang. Instead of using
# fortran language support, they are trying to find_program(amdflang).
# Since this is TheRock and we either build with our flang or no flang,
# we just hard-code this to the fortran compiler path passed because of
# FORTRAN_OPTIONAL.
# TODO: Change the project to depend on fortran properly like everything else
# does and remove this hack.
set(_rocprofsys_disable_examples "lulesh")
if(ROCM_BUILD_FORTRAN_LIBS)
  set(amdflang_EXECUTABLE "${CMAKE_Fortran_COMPILER}")
else()
  list(APPEND _rocprofsys_disable_examples "openmp-vv" "hpc")
endif()
set(ROCPROFSYS_DISABLE_EXAMPLES "${_rocprofsys_disable_examples}" CACHE STRING "" FORCE)
