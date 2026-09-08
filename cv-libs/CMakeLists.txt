# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_ENABLE_RPP)
  ##############################################################################
  # rpp (ROCm Performance Primitives) library
  ##############################################################################
  therock_cmake_subproject_declare(rpp
    USE_DIST_AMDGPU_TARGETS
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rpp"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rpp"
    COMPILER_TOOLCHAIN
      amd-hip
    RUNTIME_DEPS
      hip-clr
    BUILD_DEPS
      rocm-half
    CMAKE_ARGS
      -DROCM_PATH=
      -DBACKEND=HIP
      # TODO: bundle FFTS as a sysdep to enable audio augmentations.
      -DRPP_AUDIO_SUPPORT=OFF
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    # RPP's bundled test suite calls find_package(rpp) on itself.
    # TODO(ROCm/rocm-libraries#10163): drop once the test suite is no longer
    # configured by the library build.
    IGNORE_PACKAGES
      rpp
    INTERFACE_LINK_DIRS
      "lib"
  )
  therock_cmake_subproject_glob_c_sources(rpp
    SUBDIRS
    .
  )
  therock_cmake_subproject_provide_package(rpp rpp lib/cmake/rpp)
  therock_cmake_subproject_activate(rpp)
  therock_test_validate_shared_lib(
    PATH "${CMAKE_CURRENT_BINARY_DIR}/rpp/dist/lib"
    LIB_NAMES
      librpp.so
  )

  therock_provide_artifact(rpp
    TARGET_NEUTRAL
    DESCRIPTOR
      artifact-rpp.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      test
    SUBPROJECT_DEPS
      rpp
  )
endif(THEROCK_ENABLE_RPP)
