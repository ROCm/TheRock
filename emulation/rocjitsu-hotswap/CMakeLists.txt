# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Artifact descriptors cannot share a stage directory. This packaging-only
# subproject gives the hotswap hook its own stage without rebuilding rocjitsu.
cmake_minimum_required(VERSION 3.25)
project(rocjitsu_hotswap_package LANGUAGES NONE)

if(NOT EXISTS "${ROCJITSU_HOTSWAP_LIBRARY}")
  message(FATAL_ERROR
    "ROCJITSU_HOTSWAP_LIBRARY does not exist: ${ROCJITSU_HOTSWAP_LIBRARY}"
  )
endif()

install(FILES "${ROCJITSU_HOTSWAP_LIBRARY}" DESTINATION lib)

if(EXISTS "${ROCJITSU_HOTSWAP_TRANSLATOR_LIBRARY}")
  cmake_path(GET ROCJITSU_HOTSWAP_TRANSLATOR_LIBRARY PARENT_PATH _translator_dir)
  file(GLOB _translator_libraries
    "${_translator_dir}/librocjitsu_gfx1250_b0_to_a0.so*"
  )
  install(FILES ${_translator_libraries} DESTINATION lib)
else()
  message(WARNING
    "RocJitsu hotswap translator library is not available in this rocm-systems revision: "
    "${ROCJITSU_HOTSWAP_TRANSLATOR_LIBRARY}"
  )
endif()
