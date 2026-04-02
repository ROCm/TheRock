# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build-side targets for extended functional tests.
# Each test that needs a pre-built binary adds its subproject declaration here.

message(STATUS "extended_tests: THEROCK_ENABLE_HIP_RUNTIME=${THEROCK_ENABLE_HIP_RUNTIME} WIN32=${WIN32}")

if(NOT WIN32 AND THEROCK_ENABLE_HIP_RUNTIME)
  # Code object version backward-compatibility test (hipRTC-based).
  # Single executable that compiles a vector-add kernel at runtime via hipRTC
  # with the default code object version and the two prior versions (n-1, n-2),
  # then loads, executes, and verifies each variant.
  therock_cmake_subproject_declare(cov-backward-compat
    EXTERNAL_SOURCE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/functional/srcs"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cov-backward-compat"
    BACKGROUND_BUILD
    DISABLE_AMDGPU_TARGETS
    COMPILER_TOOLCHAIN
      amd-llvm
    RUNTIME_DEPS
      hip-clr
  )
  therock_cmake_subproject_glob_c_sources(cov-backward-compat SUBDIRS .)
  therock_cmake_subproject_activate(cov-backward-compat)

  therock_provide_artifact(core-cov-tests
    TARGET_NEUTRAL
    DESCRIPTOR
      "${CMAKE_CURRENT_SOURCE_DIR}/functional/artifact-cov-tests.toml"
    COMPONENTS
      test
    SUBPROJECT_DEPS
      cov-backward-compat
  )
endif()
