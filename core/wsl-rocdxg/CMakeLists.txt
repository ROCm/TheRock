# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This wrapper lets TheRock model the upstream libhsakmt WSL path as a normal
# subproject/artifact while the outer TheRock configure/build already runs
# inside WSL. The wrapper just drives the upstream rocdxg configure/build/install
# and stages the install tree back into TheRock's standard artifact layout.

cmake_minimum_required(VERSION 3.25)

project(therock_wsl_rocdxg NONE)

if(NOT UNIX)
  message(FATAL_ERROR "wsl-rocdxg must be configured from a Linux/WSL environment")
endif()

set(_default_rocdxg_source_dir "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocr-runtime/libhsakmt")
set(THEROCK_WSL_ROCDXG_SOURCE_DIR "${_default_rocdxg_source_dir}" CACHE PATH
  "Path to the libhsakmt source directory used for the WSL rocdxg build")
set(THEROCK_WSL_WIN_SDK "" CACHE PATH
  "Mounted Windows SDK shared include directory for the WSL rocdxg build")

if(NOT THEROCK_WSL_WIN_SDK)
  message(FATAL_ERROR "THEROCK_WSL_WIN_SDK must be set to the mounted Windows SDK shared include directory")
endif()

set(_build_root "${CMAKE_CURRENT_BINARY_DIR}/build")
set(_install_root "${CMAKE_CURRENT_BINARY_DIR}/install")
set(_stage_root "${CMAKE_CURRENT_BINARY_DIR}/stage")
set(_build_stamp "${CMAKE_CURRENT_BINARY_DIR}/wsl-rocdxg-build.stamp")
set(_build_script "${CMAKE_CURRENT_BINARY_DIR}/build_rocdxg.sh")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/wsl_build_rocdxg.sh.in"
  "${_build_script}"
  @ONLY
  NEWLINE_STYLE UNIX
)

add_custom_command(
  OUTPUT "${_build_stamp}"
  COMMAND bash "${_build_script}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${_build_stamp}"
  DEPENDS
    "${_build_script}"
    "${CMAKE_CURRENT_SOURCE_DIR}/wsl_build_rocdxg.sh.in"
  VERBATIM
  COMMENT "Build ROCDXG in WSL"
)

add_custom_target(build ALL
  DEPENDS "${_build_stamp}"
)

install(
  DIRECTORY "${_stage_root}/"
  DESTINATION .
  USE_SOURCE_PERMISSIONS
)
