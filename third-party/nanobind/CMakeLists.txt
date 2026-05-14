# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-nanobind-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/wjakob/nanobind/archive/refs/tags/v2.12.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/nanobind-2.12.0.tar.gz
  URL_HASH SHA256=01f1f0cd0398743c18f33d07ae36ad410bd7f4a1e90683b508504de897d6e629
)

therock_cmake_subproject_declare(therock-nanobind
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DNB_TEST=OFF
    -DNB_USE_SUBMODULE_DEPS=OFF
  BUILD_DEPS
    therock-robin-map
)
therock_cmake_subproject_provide_package(therock-nanobind nanobind nanobind/cmake)
therock_cmake_subproject_activate(therock-nanobind)

add_dependencies(therock-third-party therock-nanobind)
