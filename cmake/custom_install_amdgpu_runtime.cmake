# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Script run to install the amdgpu_hip_runtime component.
# This runs at install time and has access to the staging tree files.
message(STATUS "Custom installing amdgpu-runtime to: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Staging install dir: ${STAGING_INSTALL_DIR}")
if(NOT IS_DIRECTORY "${STAGING_INSTALL_DIR}")
  message(FATAL_ERROR "Did not get STAGING_INSTALL_DIR from super-project")
endif()

set(LIB_PREFIX "lib")
set(SO_SUFFIX ".so")
set(LLVM_STAGING_DIR "${STAGING_INSTALL_DIR}/llvm")

################################################################################
# Dynamic runtime files
################################################################################

# Assemble runtime dynamic file lists.
file(
  GLOB_RECURSE LIB_FILES
  LIST_DIRECTORIES FALSE
  RELATIVE ${STAGING_INSTALL_DIR}
  ${STAGING_INSTALL_DIR}/lib/*${SO_SUFFIX}
  ${STAGING_INSTALL_DIR}/lib/*${SO_SUFFIX}.*
)
list(REMOVE_ITEM LIB_FILES
  # TODO: Get the hip team to not install old version downloaded files.
  lib/libamdhip64.so.5
)

foreach(_relpath ${LIB_FILES})
  cmake_path(GET _relpath PARENT_PATH _parent_rel_path)
  file(
    INSTALL ${STAGING_INSTALL_DIR}/${_relpath}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${_parent_rel_path}
    USE_SOURCE_PERMISSIONS
  )
endforeach()

################################################################################
# LLVM runtime files
################################################################################

# NOTE: This assumes that we have linked amd_comgr statically (i.e. not
# depending on libLLVM). This makes sense for a standalone runtime. However,
# if optimizing for a full compiler+runtime, it would make sense to link against
# libLLVM and include it here.
# We collapse the comgr libraries into the runtime lib dir. The RPATHS are
# setup to make this work.
file(GLOB_RECURSE LLVM_LIB_FILES
  LIST_DIRECTORIES FALSE
  RELATIVE ${LLVM_STAGING_DIR}
  ${LLVM_STAGING_DIR}/lib/${LIB_PREFIX}amd_comgr${SO_SUFFIX}
  ${LLVM_STAGING_DIR}/lib/${LIB_PREFIX}amd_comgr${SO_SUFFIX}.*
)

foreach(_relpath ${LLVM_LIB_FILES})
  cmake_path(GET _relpath PARENT_PATH _parent_rel_path)
  file(
    INSTALL ${STAGING_INSTALL_DIR}/llvm/${_relpath}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${_parent_rel_path}
    USE_SOURCE_PERMISSIONS
  )
endforeach()
