# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# HRX's vendored IREE runtime includes the aqlprofile v2 SDK header directly.
# The core-runtime build does not otherwise build the aqlprofile project, so make
# the source SDK headers visible here without adding an artifact dependency.
set(_therock_hrx_aqlprofile_include_dir
  "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/aqlprofile/src/core/include")
set(_therock_hrx_aqlprofile_generated_include_dir
  "${CMAKE_CURRENT_BINARY_DIR}/therock-aqlprofile-include")

set(BUILD_VERSION_MAJOR 1)
set(BUILD_VERSION_MINOR 0)
set(BUILD_VERSION_PATCH 0)
set(VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}")
set(AQLPROFILE_GIT_REVISION "")
file(MAKE_DIRECTORY "${_therock_hrx_aqlprofile_generated_include_dir}/aqlprofile-sdk")
configure_file(
  "${_therock_hrx_aqlprofile_include_dir}/aqlprofile-sdk/version.h.in"
  "${_therock_hrx_aqlprofile_generated_include_dir}/aqlprofile-sdk/version.h"
  @ONLY
)

include_directories(BEFORE
  "${_therock_hrx_aqlprofile_include_dir}"
  "${_therock_hrx_aqlprofile_generated_include_dir}"
)
