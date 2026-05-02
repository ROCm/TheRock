# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_OFFLOAD_LIB_DIR)
  cmake_path(GET THEROCK_OFFLOAD_LIB_DIR PARENT_PATH _rocprofsys_offload_llvm_root)
  set(_rocprofsys_offload_bin_dir "${_rocprofsys_offload_llvm_root}/bin")

  # TheRock configures rocprofiler-systems with the amd-llvm-offload toolchain
  # when this path is set, so seed its OpenMP probes from that LLVM tree.
  foreach(_rocprofsys_compiler_name IN ITEMS amdclang++ clang++)
    if(EXISTS "${_rocprofsys_offload_bin_dir}/${_rocprofsys_compiler_name}")
      set(amdclangpp_EXECUTABLE
        "${_rocprofsys_offload_bin_dir}/${_rocprofsys_compiler_name}"
        CACHE FILEPATH "" FORCE)
      set(OMP_TARGET_COMPILER
        "${_rocprofsys_offload_bin_dir}/${_rocprofsys_compiler_name}"
        CACHE FILEPATH "" FORCE)
      break()
    endif()
  endforeach()

  find_library(LIBOMPTARGET_SO NAMES omptarget
    HINTS "${THEROCK_OFFLOAD_LIB_DIR}"
    NO_DEFAULT_PATH)

  find_library(LIBOMP_LIBRARY NAMES omp
    HINTS "${THEROCK_OFFLOAD_LIB_DIR}"
    NO_DEFAULT_PATH)
  file(GLOB _rocprofsys_openmp_include_dirs
    "${THEROCK_OFFLOAD_LIB_DIR}/clang/*/include")
  list(SORT _rocprofsys_openmp_include_dirs)
  set(_rocprofsys_openmp_include_dir)
  if(_rocprofsys_openmp_include_dirs)
    list(GET _rocprofsys_openmp_include_dirs -1 _rocprofsys_openmp_include_dir)
  endif()
  if(LIBOMP_LIBRARY AND EXISTS "${_rocprofsys_openmp_include_dir}/omp.h")
    set(OpenMP_C_FLAGS "-fopenmp=libomp" CACHE STRING "" FORCE)
    set(OpenMP_CXX_FLAGS "-fopenmp=libomp" CACHE STRING "" FORCE)
    set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
    set(OpenMP_omp_LIBRARY "${LIBOMP_LIBRARY}" CACHE FILEPATH "" FORCE)
    set(OpenMP_C_INCLUDE_DIR "${_rocprofsys_openmp_include_dir}" CACHE PATH "" FORCE)
    set(OpenMP_CXX_INCLUDE_DIR "${_rocprofsys_openmp_include_dir}" CACHE PATH "" FORCE)
    set(OpenMP_C_SPEC_DATE "201511" CACHE INTERNAL "")
    set(OpenMP_CXX_SPEC_DATE "201511" CACHE INTERNAL "")
  endif()
endif()

set(_rocprofsys_disable_examples "lulesh")
if(ROCM_BUILD_FORTRAN_LIBS)
  set(amdflang_EXECUTABLE "${CMAKE_Fortran_COMPILER}" CACHE FILEPATH "" FORCE)
  if(THEROCK_OFFLOAD_LIB_DIR)
    set(_rocprofsys_flang_module_dir "${_rocprofsys_offload_llvm_root}/include/flang")
    if(EXISTS "${_rocprofsys_flang_module_dir}/omp_lib.mod")
      set(OMPVV_FORTRAN_MODULE_DIR "${_rocprofsys_flang_module_dir}" CACHE PATH "" FORCE)
    endif()
  endif()
else()
  list(APPEND _rocprofsys_disable_examples "openmp-vv" "hpc")
endif()
set(ROCPROFSYS_DISABLE_EXAMPLES "${_rocprofsys_disable_examples}" CACHE STRING "" FORCE)
