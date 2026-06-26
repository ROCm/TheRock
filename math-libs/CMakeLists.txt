# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_ENABLE_RAND)
  ##############################################################################
  # rocRAND
  ##############################################################################
  set(_rand_subproject_names)

  therock_cmake_subproject_declare(rocRAND
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocrand"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocRAND"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_TEST=${THEROCK_BUILD_TESTING}
      -DBUILD_BENCHMARK=${THEROCK_BUILD_TESTING}
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-googletest
    RUNTIME_DEPS
      hip-clr
    TEST_SUBPROJECTS
      hipRAND
  )
  therock_cmake_subproject_glob_c_sources(rocRAND
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocRAND rocrand lib/cmake/rocrand)
  therock_cmake_subproject_activate(rocRAND)
  list(APPEND _rand_subproject_names rocRAND)

  ##############################################################################
  # hipRAND
  ##############################################################################

  therock_cmake_subproject_declare(hipRAND
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/hiprand"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipRAND"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_TEST=${THEROCK_BUILD_TESTING}
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      rocRAND
    TEST_SUBPROJECTS
  )
  therock_cmake_subproject_glob_c_sources(hipRAND
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(hipRAND hiprand lib/cmake/hiprand)
  therock_cmake_subproject_activate(hipRAND)
  list(APPEND _rand_subproject_names hipRAND)

  ##############################################################################
  # rand artifact
  ##############################################################################

  therock_provide_artifact(rand
    DESCRIPTOR artifact-rand.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS ${_rand_subproject_names}
  )
endif(THEROCK_ENABLE_RAND)

if(THEROCK_ENABLE_PRIM)
  ##############################################################################
  # rocPRIM
  ##############################################################################

  therock_cmake_subproject_declare(rocPRIM
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocprim"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocPRIM"
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_TEST=OFF
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      hip-clr
    TEST_SUBPROJECTS
      hipCUB
      rocThrust
      rocSPARSE
  )
  therock_cmake_subproject_glob_c_sources(rocPRIM
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocPRIM rocprim lib/cmake/rocprim)
  therock_cmake_subproject_activate(rocPRIM)

  if(THEROCK_BUILD_TESTING)
    ##############################################################################
    # rocPRIM tests
    ##############################################################################

    # rocPRIM_tests declared as a separate target to stop the tests from clogging up the critical build path for libs
    # that depend on rocPRIM
    # passing BACKGROUND_BUILD ensures that the tests aren't built (too) concurrently with other projects that are demanding
    # amdsmi is Linux-only and only declared when THEROCK_ENABLE_CORE_AMDSMI is ON
    # (see core/CMakeLists.txt). Benchmarks depend on primbench which requires
    # amd_smi, so they must be disabled when amdsmi is unavailable (e.g., the
    # prim-isolated CI group THEROCK_ENABLE_ALL=OFF / THEROCK_ENABLE_PRIM=ON).
    if(WIN32 OR NOT TARGET amdsmi)
      set(_rocprim_benchmark_enable "OFF")
      set(_rocprim_benchmark_deps)
      set(_rocprim_benchmark_includes)
    else()
      set(_rocprim_benchmark_enable "ON")
      set(_rocprim_benchmark_deps amdsmi)
      set(_rocprim_benchmark_includes therock_rocprim_benchmark_deps.cmake)
    endif()

    therock_cmake_subproject_declare(rocPRIM_tests
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocprim"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocPRIM_tests"
      BACKGROUND_BUILD
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DROCM_PATH=
        -DROCM_DIR=
        -DBUILD_TEST=ON
        -DBUILD_BENCHMARK=${_rocprim_benchmark_enable}
        -DUSE_SYSTEM_LIB=ON
      CMAKE_INCLUDES
        ${_rocprim_benchmark_includes}
      COMPILER_TOOLCHAIN
        amd-hip
      BUILD_DEPS
        rocm-cmake
        therock-googletest
        rocPRIM
        ${_rocprim_benchmark_deps}
      RUNTIME_DEPS
        hip-clr
    )
    therock_cmake_subproject_glob_c_sources(rocPRIM_tests
      SUBDIRS
        .
    )
    therock_cmake_subproject_activate(rocPRIM_tests)

  endif()

  therock_cmake_subproject_declare(hipCUB
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/hipcub"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipCUB"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_TEST=${THEROCK_BUILD_TESTING}
      -DBUILD_BENCHMARK=${THEROCK_BUILD_TESTING}
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      rocPRIM
      therock-googletest
    RUNTIME_DEPS
      hip-clr
  )
  therock_cmake_subproject_glob_c_sources(hipCUB
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(hipCUB hipcub lib/cmake/hipcub)
  therock_cmake_subproject_activate(hipCUB)

  therock_cmake_subproject_declare(rocThrust
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocthrust"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocThrust"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_TEST=${THEROCK_BUILD_TESTING}
      -DBUILD_BENCHMARK=${THEROCK_BUILD_TESTING}
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      rocPRIM
      rocRAND
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      ${THEROCK_BUNDLED_SQLITE3}
  )
  therock_cmake_subproject_glob_c_sources(rocThrust
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocThrust rocthrust lib/cmake/rocthrust)
  therock_cmake_subproject_activate(rocThrust)

  set(_prim_subproject_deps
    rocPRIM
    hipCUB
    rocThrust
  )

  if(THEROCK_BUILD_TESTING)
    list(APPEND _prim_subproject_deps rocPRIM_tests)
  endif()

  therock_provide_artifact(prim
    DESCRIPTOR artifact-prim.toml
    COMPONENTS
      dbg
      dev
      doc
      test
    SUBPROJECT_DEPS
      ${_prim_subproject_deps}
  )
endif(THEROCK_ENABLE_PRIM)

if(THEROCK_ENABLE_FFT)
  if(WIN32)
    # TODO(#36): enable once `rocfft_aot_helper.exe` can access hiprtc0605.dll
    #   "The code execution cannot proceed because hiprtc0605.dll was not found."
    set(_fft_kernel_cache_enable "OFF")
  else()
    set(_fft_kernel_cache_enable "ON")
  endif()

  ##############################################################################
  # rocFFT
  ##############################################################################
  set(_fft_subproject_names)

  therock_cmake_subproject_declare(rocFFT
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocfft"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocFFT"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_CLIENTS=OFF
      -DBUILD_SHARED_LIBS=ON
      -DUSE_HIPRAND=ON
      -DROCFFT_KERNEL_CACHE_ENABLE=${_fft_kernel_cache_enable}
      -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
      -DBUILD_CLIENTS_BENCH=${THEROCK_BUILD_TESTING}
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-boost
      therock-googletest
      therock-fftw3
      therock-fftw3f
    RUNTIME_DEPS
      hip-clr
      hipRAND
    TEST_SUBPROJECTS
      hipFFT
  )
  therock_cmake_subproject_glob_c_sources(rocFFT
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocFFT rocfft lib/cmake/rocfft)
  therock_cmake_subproject_activate(rocFFT)
  list(APPEND _fft_subproject_names rocFFT)

  ##############################################################################
  # hipFFT
  ##############################################################################

  therock_cmake_subproject_declare(hipFFT
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/hipfft"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipFFT"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_SHARED_LIBS=ON
      -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-boost
      therock-googletest
      therock-fftw3
      therock-fftw3f
    RUNTIME_DEPS
      hip-clr
      rocFFT
    TEST_SUBPROJECTS
  )
  therock_cmake_subproject_glob_c_sources(hipFFT
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(hipFFT hipfft lib/cmake/hipfft)
  therock_cmake_subproject_activate(hipFFT)
  list(APPEND _fft_subproject_names hipFFT)

  ##############################################################################
  # fft artifact
  ##############################################################################

  therock_provide_artifact(fft
    DESCRIPTOR artifact-fft.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS ${_fft_subproject_names}
  )
endif(THEROCK_ENABLE_FFT)

if(THEROCK_ENABLE_BLAS)
  add_subdirectory(support)
  add_subdirectory(BLAS)
endif()

if(THEROCK_ENABLE_ROCWMMA)
  ##############################################################################
  # rocWMMA
  ##############################################################################

  # Configure optional dependencies
  set(_rocwmma_optional_deps)
  if(THEROCK_ROCWMMA_ENABLE_BENCHMARKS)
    list(APPEND _rocwmma_optional_deps rocm_smi_lib)
  endif()

  therock_cmake_subproject_declare(rocWMMA
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocwmma"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocWMMA"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DROCWMMA_BUILD_TESTS=$<BOOL:${THEROCK_BUILD_TESTING}>"
      "-DROCWMMA_VALIDATE_WITH_ROCBLAS=ON"
      "-DROCWMMA_BENCHMARK_WITH_ROCBLAS=$<BOOL:${THEROCK_ROCWMMA_ENABLE_BENCHMARKS}>"
      "-DROCWMMA_BUILD_BENCHMARK_TESTS=$<BOOL:${THEROCK_ROCWMMA_ENABLE_BENCHMARKS}>"
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON # Needed for Ninja build
      -DROCWMMA_USE_SYSTEM_GOOGLETEST=ON # Use therock-googletest
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      rocBLAS
      ${_rocwmma_optional_deps}
    TEST_SUBPROJECTS
  )
  therock_cmake_subproject_glob_c_sources(rocWMMA
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocWMMA rocwmma lib/cmake/rocwmma)
  therock_cmake_subproject_activate(rocWMMA)

  therock_provide_artifact(rocwmma
    DESCRIPTOR artifact-rocwmma.toml
    COMPONENTS
      dbg
      dev
      lib
      test
    SUBPROJECT_DEPS
      rocWMMA
  )
endif(THEROCK_ENABLE_ROCWMMA)

if(THEROCK_ENABLE_ROCALUTION)
  ##############################################################################
  # rocALUTION
  ##############################################################################

  therock_cmake_subproject_declare(rocALUTION
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocalution"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocALUTION"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DSUPPORT_HIP=ON
      -DSUPPORT_MPI=OFF
      -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
      -DBUILD_CLIENTS_BENCHMARKS=OFF
      -DBUILD_CLIENTS_SAMPLES=OFF
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON # Needed for Ninja build
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      rocPRIM
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      rocBLAS
      rocSPARSE
      rocRAND
  )
  therock_cmake_subproject_glob_c_sources(rocALUTION
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocALUTION rocalution lib/cmake/rocalution)
  therock_cmake_subproject_activate(rocALUTION)

  therock_provide_artifact(rocalution
    DESCRIPTOR artifact-rocalution.toml
    COMPONENTS
      dbg
      dev
      lib
      test
    SUBPROJECT_DEPS
      rocALUTION
  )
endif(THEROCK_ENABLE_ROCALUTION)

if(THEROCK_ENABLE_LIBHIPCXX)
  ##############################################################################
  # libhipcxx
  ##############################################################################

  therock_cmake_subproject_declare(libhipcxx
    # TODO: use this in case libhipcx is moved to e.g. rocm-libraries
    #EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/libhipcxx"
    EXTERNAL_SOURCE_DIR "${THEROCK_SOURCE_DIR}/math-libs/libhipcxx"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libhipcxx"
    BACKGROUND_BUILD
    DEFAULT_GPU_TARGETS
      gfx942  # Fixes issue with empty target list for unsupported architectures.
    CMAKE_ARGS
      -DLIBHIPCXX_COPY_TO_BUILD=ON
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON # Needed for Ninja build?
      -DCMAKE_CXX_STANDARD=17
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
      hipcc
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      amd-llvm
      hip-clr
      hipcc
  )
  therock_cmake_subproject_glob_c_sources(libhipcxx
    SUBDIRS
      .
  )
  therock_cmake_subproject_require_host_program(libhipcxx lit)
  therock_cmake_subproject_activate(libhipcxx)

  therock_provide_artifact(libhipcxx
    DESCRIPTOR artifact-libhipcxx.toml
    COMPONENTS
      dbg
      dev
      lib
      test
    SUBPROJECT_DEPS
      libhipcxx
  )
endif(THEROCK_ENABLE_LIBHIPCXX)

# Most libraries need to depend on the profiler, but is conditionally only used
# on Posix.
set(optional_profiler_deps)
if(THEROCK_FLAG_INCLUDE_PROFILER)
  list(APPEND optional_profiler_deps roctracer rocprofiler-sdk)
endif()

# composable_kernel only supports a subset of gfx targets (per its own
# rocm_check_target_ids allowlist). The per-arch EXCLUDE_TARGET_PROJECTS
# mechanism in cmake/therock_amdgpu_targets.cmake removes unsupported arches
# from the CK subproject's GPU_TARGETS at activate time. If no targets remain
# after that filter (e.g. a single-arch shard for a CK-unsupported arch),
# declaring CK with an empty GPU_TARGETS list breaks CK's own HIP try-compile
# with "HIP_ARCHITECTURES is empty", so flip the global gate off here and let
# MIOpen build with MIOPEN_USE_COMPOSABLEKERNEL=OFF.
# See https://github.com/ROCm/TheRock/issues/4836
set(_ck_enabled ${THEROCK_ENABLE_COMPOSABLE_KERNEL})
if(_ck_enabled)
  therock_filter_amdgpu_targets(_ck_targets composable_kernel ${THEROCK_AMDGPU_TARGETS})
  if(NOT _ck_targets)
    message(STATUS
      "composable_kernel: no supported gfx targets remain after filtering "
      "(THEROCK_AMDGPU_TARGETS='${THEROCK_AMDGPU_TARGETS}'); skipping subproject.")
    set(_ck_enabled OFF)
  endif()
endif()
# composable_kernel now lives in math-libs but MIOpen (in ml-libs) gates its CK
# usage on _ck_enabled. math-libs is configured before ml-libs, so push the
# resolved value to the parent scope where ml-libs will inherit it.
set(_ck_enabled ${_ck_enabled} PARENT_SCOPE)

if(_ck_enabled)
  ##############################################################################
  # Composable_kernel
  ##############################################################################

  therock_cmake_subproject_declare(composable_kernel
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/composablekernel"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/composable_kernel"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DBUILD_TESTING=${THEROCK_BUILD_TESTING}"
      -DMIOPEN_REQ_LIBS_ONLY=ON
      "-DHIPTENSOR_REQ_LIBS_ONLY=$<NOT:$<BOOL:${THEROCK_COMPOSABLE_KERNEL_FOR_MIOPEN_ONLY}>>"
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      hip-clr
      rocm-half
      rocRAND
      ${THEROCK_BUNDLED_BZIP2}
      ${optional_profiler_deps}
  )
  therock_cmake_subproject_provide_package(composable_kernel composable_kernel lib/cmake/composable_kernel)

  therock_cmake_subproject_activate(composable_kernel)

  therock_provide_artifact(composable-kernel
    DESCRIPTOR artifact-composable-kernel.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
    SUBPROJECT_DEPS
      composable_kernel
  )
endif()

if(_ck_enabled AND THEROCK_ENABLE_HIPTENSOR)
  ##############################################################################
  # hipTensor
  ##############################################################################

  therock_cmake_subproject_declare(hipTensor
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/hiptensor"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipTensor"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DHIPTENSOR_BUILD_TESTS=$<BOOL:${THEROCK_BUILD_TESTING}>"
      -DHIPTENSOR_USE_SYSTEM_GOOGLETEST=ON # Use therock-googletest
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON # Needed for Ninja build
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-googletest
      composable_kernel
    RUNTIME_DEPS
      hip-clr
  )
  therock_cmake_subproject_glob_c_sources(hipTensor
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(hipTensor hiptensor lib/cmake/hiptensor)
  therock_cmake_subproject_activate(hipTensor)

  therock_provide_artifact(hiptensor
    DESCRIPTOR artifact-hiptensor.toml
    COMPONENTS
      dbg
      dev
      lib
      test
    SUBPROJECT_DEPS
      hipTensor
  )
endif(THEROCK_ENABLE_HIPTENSOR)
