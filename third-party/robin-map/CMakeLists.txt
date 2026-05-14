# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-robin-map-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/Tessil/robin-map/archive/refs/tags/v1.4.1.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/robin-map-1.4.1.tar.gz
  URL_HASH SHA256=0e3f53a377fdcdc5f9fed7a4c0d4f99e82bbb64175233bd13427fef9a771f4a1
)

therock_cmake_subproject_declare(therock-robin-map
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
)
therock_cmake_subproject_provide_package(therock-robin-map tsl-robin-map share/cmake/tsl-robin-map)
therock_cmake_subproject_activate(therock-robin-map)

add_dependencies(therock-third-party therock-robin-map)
