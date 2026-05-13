# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-robin-map-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/robin-map-1.3.0.tar.gz
  URL_HASH SHA256=a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236
)

therock_cmake_subproject_declare(therock-robin-map
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
)
therock_cmake_subproject_provide_package(therock-robin-map tsl-robin-map share/cmake/tsl-robin-map)
therock_cmake_subproject_activate(therock-robin-map)

add_dependencies(therock-third-party therock-robin-map)
