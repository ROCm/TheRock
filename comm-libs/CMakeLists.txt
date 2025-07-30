if(THEROCK_ENABLE_RCCL)
  # Most libraries need to depend on the profiler, but is conditionally only used
  # on Posix.
  set(optional_profiler_deps)
  if(NOT WIN32)
    list(APPEND optional_profiler_deps roctracer rocprofiler-sdk)
  endif()

  ##############################################################################
  # rccl
  ##############################################################################
  set(_rccl_subproject_names)

  therock_cmake_subproject_declare(rccl
    EXTERNAL_SOURCE_DIR "rccl"
    # High latency LTO link of a single library.
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DEXPLICIT_ROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
      # TODO(#56): Enable once reworked.
      -DENABLE_MSCCLPP=OFF
      -DENABLE_MSCCL_KERNEL=OFF
      -DBUILD_TESTS=${THEROCK_BUILD_TESTING}
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-fmt
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      hipify
      rocm_smi_lib
      rocprofiler-register
      ${optional_profiler_deps}
  )
  therock_cmake_subproject_glob_c_sources(rccl
    SUBDIRS
      src
  )
  therock_cmake_subproject_provide_package(rccl RCCL lib/cmake/rccl)
  therock_cmake_subproject_activate(rccl)
  list(APPEND _rccl_subproject_names rccl)

  if(THEROCK_BUILD_TESTING)
    therock_cmake_subproject_declare(rccl-tests
      EXTERNAL_SOURCE_DIR "rccl-tests"
      BACKGROUND_BUILD
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DROCM_PATH=
#        -DROCM_DIR=
        "-DEXPLICIT_ROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
        -DUSE_MPI=OFF
      COMPILER_TOOLCHAIN
        amd-hip
      BUILD_DEPS
        rocm-cmake
      RUNTIME_DEPS
        hip-clr
        hipify
        rccl
    )
    therock_cmake_subproject_glob_c_sources(rccl-tests
      SUBDIRS
        src
    )
    therock_cmake_subproject_activate(rccl-tests)
    list(APPEND _rccl_subproject_names rccl-tests)
  endif(THEROCK_BUILD_TESTING)

  therock_provide_artifact(rccl
    DESCRIPTOR artifact-rccl.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      ${_rccl_subproject_names}
  )
endif(THEROCK_ENABLE_RCCL)
