# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_ENABLE_HIPFILE)
  ##############################################################################
  # hipFile
  ##############################################################################
  set(_hipfile_subproject_names)
  therock_cmake_subproject_declare(hipfile
    USE_TEST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hipfile"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipfile"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DCMAKE_VERBOSE_MAKEFILE=OFF
      -DAIS_CXX_STANDARD=20
      -DROCM_PATH=
      -DROCM_DIR=
      -DHIP_PLATFORM=amd
      -DBUILD_TESTING=${THEROCK_BUILD_TESTING}
      -DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST
      "-DROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      ${THEROCK_BUNDLED_UTIL_LINUX}
  )
  therock_cmake_subproject_glob_c_sources(hipfile
    SUBDIRS
      src
  )
  therock_cmake_subproject_provide_package(hipfile HIPFILE lib/cmake/hipfile)
  therock_cmake_subproject_activate(hipfile)
  list(APPEND _hipfile_subproject_names hipfile)

  therock_provide_artifact(hipfile
    TARGET_NEUTRAL
    DESCRIPTOR artifact-hipfile.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      ${_hipfile_subproject_names}
      ${THEROCK_BUNDLED_UTIL_LINUX}
  )
endif(THEROCK_ENABLE_HIPFILE)
