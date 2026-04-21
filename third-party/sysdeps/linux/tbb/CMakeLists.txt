# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-tbb-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2022.3.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/oneTBB-2022.3.0.tar.gz
  URL_HASH SHA256=01598a46c1162c27253a0de0236f520fd8ee8166e9ebb84a4243574f88e6e50a
)

therock_cmake_subproject_declare(therock-tbb
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build"
  CMAKE_ARGS
    -DSOURCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/source"
    -DTBB_TEST=OFF
  INSTALL_DESTINATION
    lib/rocm_sysdeps
  INTERFACE_LINK_DIRS
    lib/rocm_sysdeps/lib
  INTERFACE_INSTALL_RPATH_DIRS
    lib/rocm_sysdeps/lib
  INTERFACE_PKG_CONFIG_DIRS
    lib/rocm_sysdeps/lib/pkgconfig
)

therock_cmake_subproject_provide_package(therock-tbb TBB lib/rocm_sysdeps/lib/cmake/TBB)
therock_cmake_subproject_activate(therock-tbb)
