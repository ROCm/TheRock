# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-frugally-deep-sources
  CMAKE_PROJECT
  # v0.21.0 (tag commit a6415158) is required by MIOpen's Keras 3 model files.
  # Note: the v0.21.0 tag still declares project(VERSION 0.20.0).
  # Originally mirrored from: https://github.com/Dobiasd/frugally-deep/archive/refs/tags/v0.21.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/frugally-deep-0.21.0.tar.gz
  URL_HASH SHA256=343bfd2120a7b8f28e90f8a63fccc3eb47089cee4a0ecf49e4dcd3136f7d4c66
)

therock_cmake_subproject_declare(therock-frugally-deep
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  BUILD_DEPS
    therock-eigen
    therock-nlohmann-json
    therock-FunctionalPlus
)
therock_cmake_subproject_provide_package(
  therock-frugally-deep frugally-deep lib/cmake/frugally-deep)
therock_cmake_subproject_activate(therock-frugally-deep)

add_dependencies(therock-third-party therock-frugally-deep)
