# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-frugally-deep-sources
  CMAKE_PROJECT
  # v0.21.1 (tag commit 9956a037) is required by MIOpen's Keras 3 model files.
  # Originally mirrored from: https://github.com/Dobiasd/frugally-deep/archive/refs/tags/v0.21.1.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/frugally-deep-0.21.1.tar.gz
  URL_HASH SHA256=0b3390eb07d0e832e81956a63b942259f9e2c2326d956042f86cb2f91a57152c
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
