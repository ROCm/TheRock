# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Shim for the legacy hip_add_executable() macro (from FindHIP.cmake).
# TheRock resolves find_package(HIP) via Config mode which doesn't define
# this macro. Sample 2_Cookbook/12_cmake_hip_add_executable needs it.
macro(hip_add_executable _target)
  cmake_parse_arguments(_args "" "" "HIPCC_OPTIONS;CLANG_OPTIONS;NVCC_OPTIONS" ${ARGN})
  set(_sources ${_args_UNPARSED_ARGUMENTS})
  list(REMOVE_ITEM _sources "EXCLUDE_FROM_ALL")
  add_executable(${_target} ${_sources})
  target_link_libraries(${_target} hip::device)
endmacro()
