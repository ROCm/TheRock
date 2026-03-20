# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Script run to install the amdgpu_runtime_dev component.
# This runs at install time and has access to the staging tree files.
message(STATUS "Custom installing amdgpu-runtime-dev to: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Staging install dir: ${STAGING_INSTALL_DIR}")
if(NOT IS_DIRECTORY "${STAGING_INSTALL_DIR}")
  message(FATAL_ERROR "Did not get STAGING_INSTALL_DIR from super-project")
endif()

file(INSTALL ${STAGING_INSTALL_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

# # Assemble file lists.
file(
  GLOB_RECURSE DEV_FILES
  LIST_DIRECTORIES FALSE
  RELATIVE ${STAGING_INSTALL_DIR}
  ${STAGING_INSTALL_DIR}/bin/hipcc_cmake_linker_helper
  ${STAGING_INSTALL_DIR}/bin/hipdemangleatp
  ${STAGING_INSTALL_DIR}/bin/roc-obj
  ${STAGING_INSTALL_DIR}/bin/roc-obj-extract
  ${STAGING_INSTALL_DIR}/bin/roc-obj-ls
  ${STAGING_INSTALL_DIR}/bin/rocm-smi
  ${STAGING_INSTALL_DIR}/lib/cmake/*
  ${STAGING_INSTALL_DIR}/lib/pkgconfig/*
  ${STAGING_INSTALL_DIR}/libexec/*
)

foreach(_relpath ${DEV_FILES})
  cmake_path(GET _relpath PARENT_PATH _parent_rel_path)
  file(
    INSTALL ${STAGING_INSTALL_DIR}/${_relpath}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${_parent_rel_path}
    USE_SOURCE_PERMISSIONS
  )
endforeach()
