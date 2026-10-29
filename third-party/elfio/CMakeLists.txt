# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

therock_subproject_fetch(therock-elfio-sources
  CMAKE_PROJECT
  # Originally mirrored from: "https://github.com/serge1/ELFIO/releases/download/Release_3.12/elfio-3.12.tar.gz"
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/elfio-3.12.tar.gz"
  URL_HASH "SHA512=13ec6be1f8ee3d7dc96d4ff141d3c199dacd818470c0d902403215571ce3d998afe9e71a8c64e0e69cfb86a4794752d93e2e406025f4812e499e51b67201e23c"
  PATCH_COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_tools/patch_third_party_source.py ${CMAKE_SOURCE_DIR}/patches/third-party/elfio
)

therock_cmake_subproject_declare(therock-elfio
  USE_DIST_AMDGPU_TARGETS
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  COMPILER_TOOLCHAIN
    amd-llvm
  CMAKE_ARGS
    -DELFIO_BUILD_EXAMPLES=OFF
    -DELFIO_BUILD_TESTS=OFF
)
therock_cmake_subproject_provide_package(therock-elfio elfio share/elfio/cmake)
therock_cmake_subproject_activate(therock-elfio)
add_dependencies(therock-third-party therock-elfio)

therock_provide_artifact(elfio
  TARGET_NEUTRAL
  DESCRIPTOR artifact-elfio.toml
  COMPONENTS
    dev
    doc
    lib
  SUBPROJECT_DEPS
    therock-elfio
)
