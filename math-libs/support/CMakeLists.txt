# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

################################################################################
# mxDataGenerator
################################################################################

therock_cmake_subproject_declare(mxDataGenerator
  FPRINT_PROJECT_SOURCE
  FPRINT_PROJECT_OPTIONAL_INPUTS ${THEROCK_PROJECT_ROOT_INPUTS}
  FPRINT_PROJECT_ENVIRONMENT
    ${THEROCK_PROJECT_ROCM_VERSION_ENVIRONMENT} ${THEROCK_PROJECT_PACKAGE_ENVIRONMENT}
    ${THEROCK_PROJECT_LIBRARY_ENVIRONMENT}
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/shared/mxdatagenerator"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/mxDataGenerator"
  BACKGROUND_BUILD
  CMAKE_ARGS
    # TODO: Enable tests once there is install support
    #-DMXDATAGENERATOR_BUILD_TESTING=${THEROCK_BUILD_TESTING}
  COMPILER_TOOLCHAIN
    amd-hip
  BUILD_DEPS
    amd-llvm
    therock-googletest
)
therock_cmake_subproject_provide_package(mxDataGenerator
  mxDataGenerator lib/cmake)
therock_cmake_subproject_activate(mxDataGenerator)

################################################################################
# support artifact
################################################################################

therock_provide_artifact(support
  TARGET_NEUTRAL
  DESCRIPTOR artifact-support.toml
  COMPONENTS
    dev
    doc
  SUBPROJECT_DEPS
    mxDataGenerator
)
