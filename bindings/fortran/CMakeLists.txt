# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_ENABLE_RUNTIME_FORTRAN)
  ##############################################################################
  # runtime-fortran: HIP and ROCTx
  ##############################################################################
  set(_runtime_fortran_subproject_names)
  if(NOT THEROCK_SANITIZER AND EXISTS "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/bindings/fortran/hip/CMakeLists.txt")
    # The HIP tests build a code object with hipcc, which does not know where
    # amd-llvm's device libraries are.
    therock_compiler_toolchain_subproject(_toolchain_subproject amd-hip)
    get_target_property(_amd_llvm_dist_dir "${_toolchain_subproject}" THEROCK_DIST_DIR)
    therock_cmake_subproject_declare(hip-fortran
      USE_TEST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/bindings/fortran/hip"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran"
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DBUILD_FORTRAN_BINDINGS=ON
        -DBUILD_FORTRAN_CLIENTS=${THEROCK_BUILD_TESTING}
        "-DHIP_FORTRAN_GENCO_FLAGS=--hip-device-lib-path=${_amd_llvm_dist_dir}/lib/llvm/amdgcn/bitcode"
      COMPILER_TOOLCHAIN
        amd-hip
      RUNTIME_DEPS
        hip-clr
    )
    therock_cmake_subproject_provide_package(hip-fortran hip-fortran lib/cmake/hip-fortran)
    therock_cmake_subproject_activate(hip-fortran)
    list(APPEND _runtime_fortran_subproject_names hip-fortran)
  endif()

  # ROCTx is host-only: no HIP compiler, device code or GPU targets.
  if(NOT THEROCK_SANITIZER AND EXISTS "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/bindings/fortran/roctx/CMakeLists.txt")
    therock_cmake_subproject_declare(roctx-fortran
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/bindings/fortran/roctx"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran"
      CMAKE_ARGS
        -DBUILD_FORTRAN_BINDINGS=ON
        -DBUILD_FORTRAN_CLIENTS=${THEROCK_BUILD_TESTING}
      RUNTIME_DEPS
        rocprofiler-sdk
    )
    therock_cmake_subproject_provide_package(roctx-fortran roctx-fortran lib/cmake/roctx-fortran)
    therock_cmake_subproject_activate(roctx-fortran)
    list(APPEND _runtime_fortran_subproject_names roctx-fortran)
  endif()

  therock_provide_artifact(runtime-fortran
    TARGET_NEUTRAL
    DESCRIPTOR artifact-runtime-fortran.toml
    COMPONENTS
      dev
      test
    SUBPROJECT_DEPS
      ${_runtime_fortran_subproject_names}
  )
endif(THEROCK_ENABLE_RUNTIME_FORTRAN)

# The math-library binding tests `use hip`, so they need hip-fortran, but not
# roctx-fortran.
set(_hip_fortran_dep)
if(TARGET hip-fortran)
  set(_hip_fortran_dep hip-fortran)
endif()

function(therock_declare_library_fortran_binding out_var lib)
  cmake_parse_arguments(PARSE_ARGV 2 ARG "" "" "RUNTIME_DEPS;PACKAGES")
  set(_source_dir "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/bindings/fortran/${lib}")
  if(THEROCK_SANITIZER OR NOT EXISTS "${_source_dir}/CMakeLists.txt")
    return()
  endif()
  if(NOT ARG_PACKAGES)
    set(ARG_PACKAGES "${lib}-fortran")
  endif()
  set(_name "${lib}-fortran")
  therock_cmake_subproject_declare(${_name}
    USE_TEST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${_source_dir}"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${_name}"
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DBUILD_FORTRAN_BINDINGS=ON
      -DBUILD_FORTRAN_CLIENTS=${THEROCK_BUILD_TESTING}
    COMPILER_TOOLCHAIN
      amd-hip
    RUNTIME_DEPS
      ${ARG_RUNTIME_DEPS}
      ${_hip_fortran_dep}
  )
  foreach(_package IN LISTS ARG_PACKAGES)
    therock_cmake_subproject_provide_package(${_name} ${_package} lib/cmake/${_package})
  endforeach()
  therock_cmake_subproject_activate(${_name})
  set(${out_var} ${${out_var}} ${_name} PARENT_SCOPE)
endfunction()

if(THEROCK_ENABLE_BLAS_FORTRAN)
  ##############################################################################
  # blas-fortran: rocBLAS and hipBLAS
  ##############################################################################
  set(_blas_fortran_subproject_names)
  therock_declare_library_fortran_binding(_blas_fortran_subproject_names rocblas
    RUNTIME_DEPS rocBLAS)
  therock_declare_library_fortran_binding(_blas_fortran_subproject_names hipblas
    RUNTIME_DEPS hipBLAS hipBLAS-common)

  therock_provide_artifact(blas-fortran
    TARGET_NEUTRAL
    DESCRIPTOR artifact-blas-fortran.toml
    COMPONENTS
      dev
      test
    SUBPROJECT_DEPS
      ${_blas_fortran_subproject_names}
  )
endif(THEROCK_ENABLE_BLAS_FORTRAN)

if(THEROCK_ENABLE_SOLVER_FORTRAN)
  ##############################################################################
  # solver-fortran: rocSOLVER and hipSOLVER
  ##############################################################################
  set(_solver_fortran_subproject_names)
  set(_rocblas_fortran_dep)
  if(TARGET rocblas-fortran)
    set(_rocblas_fortran_dep rocblas-fortran)
  endif()
  therock_declare_library_fortran_binding(_solver_fortran_subproject_names rocsolver
    RUNTIME_DEPS rocSOLVER ${_rocblas_fortran_dep})
  therock_declare_library_fortran_binding(_solver_fortran_subproject_names hipsolver
    RUNTIME_DEPS hipSOLVER)

  therock_provide_artifact(solver-fortran
    TARGET_NEUTRAL
    DESCRIPTOR artifact-solver-fortran.toml
    COMPONENTS
      dev
      test
    SUBPROJECT_DEPS
      ${_solver_fortran_subproject_names}
  )
endif(THEROCK_ENABLE_SOLVER_FORTRAN)

if(THEROCK_ENABLE_SPARSE_FORTRAN)
  ##############################################################################
  # sparse-fortran: rocSPARSE and hipSPARSE
  ##############################################################################
  set(_sparse_fortran_subproject_names)
  therock_declare_library_fortran_binding(_sparse_fortran_subproject_names rocsparse
    RUNTIME_DEPS rocSPARSE)
  therock_declare_library_fortran_binding(_sparse_fortran_subproject_names hipsparse
    RUNTIME_DEPS hipSPARSE)

  therock_provide_artifact(sparse-fortran
    TARGET_NEUTRAL
    DESCRIPTOR artifact-sparse-fortran.toml
    COMPONENTS
      dev
      test
    SUBPROJECT_DEPS
      ${_sparse_fortran_subproject_names}
  )
endif(THEROCK_ENABLE_SPARSE_FORTRAN)

if(THEROCK_ENABLE_FFT_FORTRAN)
  ##############################################################################
  # fft-fortran: rocFFT, and hipFFT with hipFFTW
  ##############################################################################
  set(_fft_fortran_subproject_names)
  therock_declare_library_fortran_binding(_fft_fortran_subproject_names rocfft
    RUNTIME_DEPS rocFFT)
  therock_declare_library_fortran_binding(_fft_fortran_subproject_names hipfft
    RUNTIME_DEPS hipFFT
    PACKAGES hipfft-fortran hipfftw-fortran)

  therock_provide_artifact(fft-fortran
    TARGET_NEUTRAL
    DESCRIPTOR artifact-fft-fortran.toml
    COMPONENTS
      dev
      test
    SUBPROJECT_DEPS
      ${_fft_fortran_subproject_names}
  )
endif(THEROCK_ENABLE_FFT_FORTRAN)

if(THEROCK_ENABLE_RAND_FORTRAN)
  ##############################################################################
  # rand-fortran: rocRAND and hipRAND
  ##############################################################################
  set(_rand_fortran_subproject_names)
  therock_declare_library_fortran_binding(_rand_fortran_subproject_names rocrand
    RUNTIME_DEPS rocRAND)
  therock_declare_library_fortran_binding(_rand_fortran_subproject_names hiprand
    RUNTIME_DEPS hipRAND)

  therock_provide_artifact(rand-fortran
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rand-fortran.toml
    COMPONENTS
      dev
      test
    SUBPROJECT_DEPS
      ${_rand_fortran_subproject_names}
  )
endif(THEROCK_ENABLE_RAND_FORTRAN)
