# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-nanobind-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/wjakob/nanobind/archive/refs/tags/v2.4.0.tar.gz
  URL https://github.com/wjakob/nanobind/archive/refs/tags/v2.4.0.tar.gz
  URL_HASH SHA256=bb35deaed7efac5029ed1e33880a415638352f757d49207a8e6013fefb6c49a7
)

# nanobind bundles robin_map as a git submodule which GitHub tarballs exclude.
# Fetch it separately into nanobind's ext/robin_map directory.
therock_subproject_fetch(therock-nanobind-robin-map-sources
  URL https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz
  URL_HASH SHA256=a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source/ext/robin_map"
  TOUCH "${CMAKE_CURRENT_BINARY_DIR}/source/ext/robin_map/include/tsl/robin_map.h"
)
add_dependencies(therock-nanobind-robin-map-sources therock-nanobind-sources)

therock_cmake_subproject_declare(therock-nanobind
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  EXTRA_DEPENDS therock-nanobind-robin-map-sources
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
