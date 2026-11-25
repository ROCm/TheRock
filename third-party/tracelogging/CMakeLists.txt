# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# TraceLogging for LTTng: the TraceLoggingProvider.h headers and the lttngh
# static library, used by the HIP and HSA runtimes to write their API trace
# events through liblttng-ust. Only the LTTng/ project of the repository is
# built; on Windows the same headers come from the Windows SDK.
therock_subproject_fetch(therock-tracelogging-sources
  # The merge commit of microsoft/tracelogging#85 (installs the find modules the
  # package config needs), which TheRock used to apply as a patch.
  # Originally from: https://github.com/microsoft/tracelogging/archive/d8e68c949160e63bde71fc5c815dffdc21c9a358.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/tracelogging-d8e68c949160e63bde71fc5c815dffdc21c9a358.tar.gz
  URL_HASH SHA256=c0816a38517f81422768027355ba9e7277320cf7d20776c618dd614e2f9c2557
  # The CMake project is in LTTng/, not at the root (as CMAKE_PROJECT assumes):
  # mark its CMakeLists.txt as the fetch's by-product instead.
  TOUCH "${CMAKE_CURRENT_BINARY_DIR}/source/LTTng/CMakeLists.txt"
)

therock_cmake_subproject_declare(therock-tracelogging
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source/LTTng"
  # The fetched tree's fingerprint is recorded next to source/, not source/LTTng.
  FPRINT_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  RUNTIME_DEPS
    ${THEROCK_BUNDLED_LTTNG_UST}
)
therock_cmake_subproject_provide_package(
  therock-tracelogging tracelogging lib/cmake/tracelogging)
therock_cmake_subproject_activate(therock-tracelogging)

therock_provide_artifact(tracelogging
  TARGET_NEUTRAL
  DESCRIPTOR artifact-tracelogging.toml
  COMPONENTS
    dev
  SUBPROJECT_DEPS
    therock-tracelogging
)

add_dependencies(therock-third-party therock-tracelogging)
