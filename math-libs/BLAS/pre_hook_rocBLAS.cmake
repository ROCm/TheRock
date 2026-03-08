# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Tensile just uses the system path to find most of its tools and it does this
# in the build phase. Rather than tunneling everything through manually, we
# just explicitly set up the path to include our toolchain ROCM and LLVM
# tools. This kind of reacharound is not great but the project is old, so
# c'est la vie.

block(SCOPE_FOR VARIABLES)
  if(NOT THEROCK_TOOLCHAIN_ROOT)
    message(FATAL_ERROR "As a sub-project, THEROCK_TOOLCHAIN_ROOT should have been defined and was not")
  endif()
  if(WIN32)
    set(PS ";")
  else()
    set(PS ":")
  endif()
  set(CURRENT_PATH "$ENV{PATH}")
  set(ENV{PATH} "${THEROCK_TOOLCHAIN_ROOT}/bin${PS}${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin${PS}${CURRENT_PATH}")
  message(STATUS "Augmented toolchain PATH=$ENV{PATH}")
endblock()

# Tensile is using msgpack and will pull in Boost otherwise.
add_compile_definitions(MSGPACK_NO_BOOST)

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

if(NOT WIN32)
  # Configure roctracer if on a supported operating system (Linux).
  # rocBLAS has deprecated dependencies on roctracer. We apply a patch to redirect
  # naked linking against `-lroctx64` to an explicitly found version of the library.
  # See: https://github.com/ROCm/TheRock/issues/364
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
  find_library(_therock_legacy_roctx64 roctx64 REQUIRED)
  cmake_language(DEFER CALL therock_patch_linked_lib OLD_LIBRARY "roctx64" NEW_TARGET "${_therock_legacy_roctx64}")
endif()
