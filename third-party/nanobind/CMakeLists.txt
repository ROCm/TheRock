# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-nanobind-sources
  CMAKE_PROJECT
  # Bundled tarball of nanobind v2.4.0 with robin_map v1.3.0 (git submodule).
  # Originally mirrored from: https://github.com/wjakob/nanobind/archive/refs/tags/v2.4.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/nanobind-2.4.0.tar.gz
  URL_HASH SHA256=df475b22680e3be1729ca322616ff05b96fd20bf8f85f736e952b71235c09664
)

therock_cmake_subproject_declare(therock-nanobind
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DNB_TEST=OFF
)
therock_cmake_subproject_provide_package(therock-nanobind nanobind nanobind/cmake)
therock_cmake_subproject_activate(therock-nanobind)

therock_provide_artifact(nanobind
  TARGET_NEUTRAL
  DESCRIPTOR artifact-nanobind.toml
  COMPONENTS
    dev
  SUBPROJECT_DEPS
    therock-nanobind
)

add_dependencies(therock-third-party therock-nanobind)
