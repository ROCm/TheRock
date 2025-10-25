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
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-googletest
    RUNTIME_DEPS
      hip-clr
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
  )
  therock_cmake_subproject_glob_c_sources(rocPRIM
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocPRIM rocprim lib/cmake/rocprim)
  therock_cmake_subproject_activate(rocPRIM)

  therock_cmake_subproject_declare(hipCUB
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/hipcub"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipCUB"
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
      rocPRIM
      therock-googletest
    RUNTIME_DEPS
      hip-clr
  )
  therock_cmake_subproject_glob_c_sources(rocPRIM
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
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      rocPRIM
      therock-googletest
    RUNTIME_DEPS
      hip-clr
  )
  therock_cmake_subproject_glob_c_sources(rocThrust
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocThrust rocthrust lib/cmake/rocthrust)
  therock_cmake_subproject_activate(rocThrust)

  therock_provide_artifact(prim
    DESCRIPTOR artifact-prim.toml
    COMPONENTS
      dbg
      dev
      doc
      test
    SUBPROJECT_DEPS rocPRIM hipCUB rocThrust
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

if(NOT WIN32) # rocWMMA is disabled on Windows: The detected ROCm does not support data type `__hip_fp8_e5m2` or `__hip_fp8_e4m3`.
  set(_rocWMMA_optional_runtime_deps)
  set(_rocWMMA_use_rocBLAS OFF)

  if(THEROCK_ENABLE_BLAS AND THEROCK_BUILD_TESTING)
    set(_rocWMMA_optional_runtime_deps rocBLAS)
    set(_rocWMMA_use_rocBLAS ON)
  endif()

  therock_cmake_subproject_declare(rocWMMA
      EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocWMMA"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocWMMA"
      CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCWMMA_BUILD_TESTS=${THEROCK_BUILD_TESTING}
      -DROCWMMA_BUILD_SAMPLES=OFF # memory-hungry
      -DROCWMMA_BUILD_ASSEMBLY=OFF
      -DBUILD_OFFLOAD_COMPRESS=OFF
      -DROCWMMA_CODE_COVERAGE=OFF # clang only
      -DROCWMMA_BUILD_VALIDATION_TESTS=${THEROCK_BUILD_TESTING}
      -DROCWMMA_BUILD_BENCHMARK_TESTS=${THEROCK_BUILD_TESTING}
      -DROCWMMA_BUILD_EXTENDED_TESTS=${THEROCK_BUILD_TESTING}
      -DROCWMMA_VALIDATE_WITH_ROCBLAS=${_rocWMMA_use_rocBLAS}
      -DROCWMMA_BENCHMARK_WITH_ROCBLAS=${_rocWMMA_use_rocBLAS}
      -DUSE_SYSTEM_GOOGLETEST=ON
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
      COMPILER_TOOLCHAIN
        amd-hip
      BUILD_DEPS
        therock-googletest
        rocm-cmake
      RUNTIME_DEPS
        hip-clr
        ${_rocWMMA_optional_runtime_deps}
    )
    therock_cmake_subproject_glob_c_sources(rocFFT
      SUBDIRS
        .
    )
    therock_cmake_subproject_provide_package(rocWMMA rocWMMA lib/cmake/rocWMMA)
    therock_cmake_subproject_activate(rocWMMA)
endif()
