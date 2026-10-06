# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

################################################################################
# mxDataGenerator
################################################################################

set(_mxdatagenerator_enabled ON)
if(THEROCK_AMDGPU_TARGETS AND NOT "${THEROCK_AMDGPU_TARGETS}" MATCHES "-NOTFOUND$")
  therock_filter_amdgpu_targets(_mxdatagenerator_targets mxDataGenerator ${THEROCK_AMDGPU_TARGETS})
  if(NOT _mxdatagenerator_targets)
    message(STATUS
      "mxDataGenerator: no supported gfx targets remain after filtering "
      "(THEROCK_AMDGPU_TARGETS='${THEROCK_AMDGPU_TARGETS}'); skipping subproject.")
    set(_mxdatagenerator_enabled OFF)
  endif()
endif()

if(_mxdatagenerator_enabled)
  therock_cmake_subproject_declare(mxDataGenerator
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
endif()
