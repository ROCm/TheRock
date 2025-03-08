if(THEROCK_ENABLE_RAND)
  ##############################################################################
  # rocRAND
  ##############################################################################
  set(_rand_subproject_names)

  therock_cmake_subproject_declare(rocRAND
    EXTERNAL_SOURCE_DIR "rocRAND"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
    COMPILER_TOOLCHAIN
      amd-hip
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
    EXTERNAL_SOURCE_DIR "hipRAND"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocRAND
    RUNTIME_DEPS
      hip-clr
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
    SUBPROJECT_DEPS ${_rand_subproject_names}
  )
endif(THEROCK_ENABLE_RAND)

if(THEROCK_ENABLE_PRIM)
  ##############################################################################
  # rocPRIM
  ##############################################################################

  therock_cmake_subproject_declare(rocPRIM
    EXTERNAL_SOURCE_DIR "rocPRIM"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
    COMPILER_TOOLCHAIN
      amd-hip
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
    EXTERNAL_SOURCE_DIR "hipCUB"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
    COMPILER_TOOLCHAIN
      amd-hip
    RUNTIME_DEPS
      hip-clr
    BUILD_DEPS
      rocPRIM
  )
  therock_cmake_subproject_glob_c_sources(rocPRIM
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(hipCUB hipcub lib/cmake/hipcub)
  therock_cmake_subproject_activate(hipCUB)

  therock_cmake_subproject_declare(rocThrust
    EXTERNAL_SOURCE_DIR "rocThrust"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_TEST=ON
    COMPILER_TOOLCHAIN
      amd-hip
    RUNTIME_DEPS
      hip-clr
    BUILD_DEPS
      rocPRIM
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
      dev
      doc
      test
    SUBPROJECT_DEPS rocPRIM hipCUB rocThrust
  )
endif(THEROCK_ENABLE_PRIM)

if(THEROCK_ENABLE_FFT)
  ##############################################################################
  # rocFFT
  ##############################################################################
  set(_fft_subproject_names)

  therock_cmake_subproject_declare(rocFFT
    EXTERNAL_SOURCE_DIR "rocFFT"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_CLIENTS=OFF
      -DUSE_HIPRAND=ON
    COMPILER_TOOLCHAIN
      amd-hip
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
    EXTERNAL_SOURCE_DIR "hipFFT"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
    COMPILER_TOOLCHAIN
      amd-hip
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
    SUBPROJECT_DEPS ${_fft_subproject_names}
  )
endif(THEROCK_ENABLE_FFT)

if(THEROCK_ENABLE_BLAS)
  add_subdirectory(BLAS)
endif()
