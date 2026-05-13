# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-nanobind-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/wjakob/nanobind/archive/refs/tags/v2.4.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/nanobind-2.4.0.tar.gz
  URL_HASH SHA256=bb35deaed7efac5029ed1e33880a415638352f757d49207a8e6013fefb6c49a7
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
)
therock_cmake_subproject_provide_package(therock-nanobind nanobind nanobind/cmake)
therock_cmake_subproject_activate(therock-nanobind)

add_dependencies(therock-third-party therock-nanobind)
