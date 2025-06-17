if(THEROCK_ENABLE_MIOPEN)
  # Most libraries need to depend on the profiler, but is conditionally only used
  # on Posix.
  set(optional_profiler_deps)
  if(NOT WIN32)
    list(APPEND optional_profiler_deps roctracer rocprofiler-sdk)
  endif()

  ##############################################################################
  # Composable_kernel
  ##############################################################################

  therock_cmake_subproject_declare(composable_kernel
    EXTERNAL_SOURCE_DIR "composable_kernel"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DBUILD_TESTING=${THEROCK_BUILD_TESTING}"
      -DMIOPEN_REQ_LIBS_ONLY=ON
      -DCMAKE_BUILD_TYPE=Release
      -DGPU_TARGETS="gfx90a"
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      hipBLAS-common
      hip-clr
      hipBLAS
      hipBLASLt
      rocm-half
      rocBLAS
      rocRAND
      ${THEROCK_BUNDLED_BZIP2}
      ${THEROCK_BUNDLED_SQLITE3}
      ${optional_profiler_deps}
  )
  therock_cmake_subproject_provide_package(composable_kernel libdevice_conv_operations lib/libdevice_conv_operations)
  therock_cmake_subproject_provide_package(composable_kernel libdevice_other_operations lib/libdevice_other_operations)

  therock_cmake_subproject_activate(composable_kernel)

  therock_provide_artifact(composable_kernel
    DESCRIPTOR artifact-composable_kernel.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      composable_kernel
  )

  ##############################################################################
  # MIOpen
  ##############################################################################

  if(MSVC)
    # HACK, see https://github.com/ROCm/TheRock/issues/525.
    # Take the most recent few MSVC versions from
    # https://github.com/boostorg/boost_install/blob/develop/BoostDetectToolset.cmake
    # and pass them through to the Boost_COMPILER option below. This is messy -
    # maybe we can drop the Boost dependency or write new build rules for it.
    if((MSVC_VERSION GREATER 1929) AND (MSVC_VERSION LESS 1950))
      set(_THEROCK_MIOPEN_OVERRIDE_BOOST_COMPILER "vc143")
    elseif((MSVC_VERSION GREATER 1919) AND (MSVC_VERSION LESS 1930))
      set(_THEROCK_MIOPEN_OVERRIDE_BOOST_COMPILER "vc142")
    elseif((MSVC_VERSION GREATER 1909) AND (MSVC_VERSION LESS 1920))
      set(_THEROCK_MIOPEN_OVERRIDE_BOOST_COMPILER "vc141")
    else()
      message(WARNING "Unhandled MSVC_VERSION '${MSVC_VERSION}', Boost may not configure correctly")
    endif()
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
      -DBoost_COMPILER=${_THEROCK_MIOPEN_OVERRIDE_BOOST_COMPILER}
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-boost
      therock-eigen
      therock-frugally-deep
      therock-googletest
      therock-nlohmann-json
      therock-FunctionalPlus
    RUNTIME_DEPS
      hipBLAS-common
      hip-clr
      hipBLAS
      hipBLASLt
      rocm-half
      rocBLAS
      rocRAND
      ${THEROCK_BUNDLED_BZIP2}
      ${THEROCK_BUNDLED_SQLITE3}
      ${optional_profiler_deps}
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
