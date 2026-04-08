# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This finder resolves the virtual BLAS package for sub-projects.
# It defers to the built host-blas, if available, otherwise, failing.
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

if("OpenBLAS" IN_LIST THEROCK_PROVIDED_PACKAGES)
  cmake_policy(POP)
  message(STATUS "Resolving bundled host-blas library from super-project")

  if(DEFINED BLA_SIZEOF_INTEGER AND BLA_SIZEOF_INTEGER EQUAL 8)
    find_package(OpenBLAS64 CONFIG REQUIRED)
    set(_OPENBLAS OpenBLAS64)
  else()
    find_package(OpenBLAS CONFIG REQUIRED)
    set(_OPENBLAS OpenBLAS)
  endif()

  # OpenBLAS may export OpenMP::OpenMP_C as an interface dependency when built
  # with USE_OPENMP=ON, but consumers only need the already-linked libopenblas.so.
  # Remove it from the interface so CXX-only projects don't need C OpenMP setup.
  if(TARGET ${_OPENBLAS}::OpenBLAS)
    get_target_property(_openblas_iface_libs ${_OPENBLAS}::OpenBLAS INTERFACE_LINK_LIBRARIES)
    if(_openblas_iface_libs)
      list(REMOVE_ITEM _openblas_iface_libs "OpenMP::OpenMP_C")
      set_target_properties(${_OPENBLAS}::OpenBLAS PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_openblas_iface_libs}")
    endif()
  endif()

  # See: https://cmake.org/cmake/help/latest/module/FindBLAS.html
  set(BLAS_LINKER_FLAGS)
  set(BLAS_LIBRARIES ${_OPENBLAS}::OpenBLAS)
  add_library(BLAS::BLAS ALIAS ${_OPENBLAS}::OpenBLAS)
  set(BLAS95_LIBRARIES)
  set(BLAS95_FOUND FALSE)
  set(BLAS_FOUND TRUE)
else()
  cmake_policy(POP)
  set(BLAS_FOUND FALSE)
endif()
