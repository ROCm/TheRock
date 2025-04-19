if(THEROCK_ENABLE_MIOPEN)
  ##############################################################################
  # MIOpen
  ##############################################################################

  if(WIN32)
    set(THEROCK_OVERRIDE_BOOST_COMPILER vc143)
  endif()

  therock_cmake_subproject_declare(MIOpen
    EXTERNAL_SOURCE_DIR "MIOpen"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DBUILD_TESTING=${THEROCK_BUILD_TESTING}"
      -DMIOPEN_USE_COMPOSABLEKERNEL=OFF  # TODO: enable
      -DMIOPEN_USE_MLIR=OFF # TODO: enable
      -DMIOPEN_BUILD_DRIVER=ON
      -DMIOPEN_USE_ROCTRACER=OFF # TODO: enable
      -DMIOPEN_USE_HIPBLASLT=ON
      -DBoost_COMPILER=${THEROCK_OVERRIDE_BOOST_COMPILER}
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      hipBLAS-common
      rocm-cmake
      rocm-half
      therock-boost
      therock-eigen
      therock-frugally-deep
      therock-googletest
      therock-nlohmann-json
      therock-FunctionalPlus
    RUNTIME_DEPS
      hip-clr
      hipBLAS
      hipBLASLt
      rocBLAS
      rocRAND
      ${THEROCK_BUNDLED_BZIP2}
      ${THEROCK_BUNDLED_SQLITE3}
  )

  therock_cmake_subproject_activate(MIOpen)

  therock_provide_artifact(miopen
    DESCRIPTOR artifact-miopen.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      MIOpen
  )
endif(THEROCK_ENABLE_MIOPEN)
