if(THEROCK_ENABLE_RCCL)
  ##############################################################################
  # rccl
  ##############################################################################

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
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      hip-clr
      hipify
      rocm_smi_lib
      rocprofiler-register
  )
  therock_cmake_subproject_glob_c_sources(rccl
    SUBDIRS
      src
  )
  therock_cmake_subproject_activate(rccl)

  therock_provide_artifact(rccl
    DESCRIPTOR artifact-rccl.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      rccl
  )
endif(THEROCK_ENABLE_RCCL)
