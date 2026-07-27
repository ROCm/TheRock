# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Artifact descriptors cannot share a stage directory. This packaging-only
# subproject gives the hotswap hook its own stage without rebuilding rocjitsu.
cmake_minimum_required(VERSION 3.25)
project(rocjitsu_hotswap_package LANGUAGES NONE)

if(NOT THEROCK_STAGE_INSTALL_ROOT)
  message(FATAL_ERROR "THEROCK_STAGE_INSTALL_ROOT is required")
endif()

if(NOT EXISTS "${ROCJITSU_HOTSWAP_LIBRARY}")
  message(FATAL_ERROR
    "ROCJITSU_HOTSWAP_LIBRARY does not exist: ${ROCJITSU_HOTSWAP_LIBRARY}"
  )
endif()

install(FILES "${ROCJITSU_HOTSWAP_LIBRARY}" DESTINATION lib)
