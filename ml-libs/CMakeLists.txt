# Most libraries need to depend on the profiler, but is conditionally only used
# on Posix.
set(optional_profiler_deps)
if(NOT WIN32)
  list(APPEND optional_profiler_deps roctracer rocprofiler-sdk)
endif()

if(THEROCK_ENABLE_COMPOSABLE_KERNEL)
  ##############################################################################
  # Composable_kernel
  ##############################################################################
  # TODO: Move this to math-libs

  therock_cmake_subproject_declare(composable_kernel
    EXTERNAL_SOURCE_DIR "${THEROCK_COMPOSABLE_KERNEL_SOURCE_DIR}"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/composable_kernel"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DBUILD_TESTING=${THEROCK_BUILD_TESTING}"
      -DMIOPEN_REQ_LIBS_ONLY=ON
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
      run
    SUBPROJECT_DEPS
      composable_kernel
  )
endif()

if(THEROCK_ENABLE_MIOPEN)
  ##############################################################################
  # MIOpen
  ##############################################################################

  # composable kernel is currently only tested and gfx908, gfx90a, gfx942 and
  # gfx950. For other GPU families, building MIOpen with CK enabled, the build is
  # failing with linker errors which need further investigation.
  if(THEROCK_MIOPEN_USE_COMPOSABLE_KERNEL)
    set(_ck_supported_gfx_targets gfx908 gfx90a gfx942 gfx950)
    foreach(_gfx_target ${THEROCK_AMDGPU_TARGETS})
      if(NOT ${_gfx_target} IN_LIST _ck_supported_gfx_targets)
        message(WARNING
          "${_gfx_target} (included in THEROCK_AMDGPU_TARGETS) is not supported by composable kernel. "
          "Disabling using composable kernel in MIOpen.")
        set(THEROCK_MIOPEN_USE_COMPOSABLE_KERNEL OFF)
      endif()
    endforeach()
  endif()

  # If composable kernel is enabled, add it as an MIOpen dep.
  set(optional_miopen_build_deps)
  if(THEROCK_MIOPEN_USE_COMPOSABLE_KERNEL)
    list(APPEND optional_miopen_build_deps composable_kernel)
  endif()

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

    # TODO(#1407): re-enable MIOpen tests on Windows when they don't crash clang on CI builders
    #   We also don't run MIOpen tests on Windows, see
    #   build_tools/github_actions/fetch_test_configurations.py
    set(_THEROCK_MIOPEN_ENABLE_TESTING OFF)
  else()
    set(_THEROCK_MIOPEN_ENABLE_TESTING ${THEROCK_BUILD_TESTING})
  endif()

  therock_cmake_subproject_declare(MIOpen
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/miopen"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/MIOpen"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DBUILD_TESTING=${_THEROCK_MIOPEN_ENABLE_TESTING}"
      "-DMIOPEN_USE_COMPOSABLEKERNEL=${THEROCK_MIOPEN_USE_COMPOSABLE_KERNEL}"
      -DMIOPEN_USE_MLIR=OFF # TODO: enable
      -DMIOPEN_TEST_DISCRETE=OFF
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
      ${optional_miopen_build_deps}
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
      test
    SUBPROJECT_DEPS
      MIOpen
  )
endif(THEROCK_ENABLE_MIOPEN)
