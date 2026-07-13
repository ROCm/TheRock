# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_debug_info.cmake
# Resolution of debug-info settings (THEROCK_GENERATE_DEBUG_INFO,
# THEROCK_DEBUG_INFO_LEVEL, THEROCK_SPLIT_DEBUG_INFO) into effective per-subproject
# (or global baseline) values.

function(_therock_normalize_bool out_var value context)
  string(TOUPPER "${value}" _upper)
  if(_upper MATCHES "^(ON|TRUE|YES|Y|1)$")
    set(${out_var} ON PARENT_SCOPE)
  elseif(_upper MATCHES "^(OFF|FALSE|NO|N|0)$")
    set(${out_var} OFF PARENT_SCOPE)
  else()
    message(FATAL_ERROR "${context} must be a boolean (ON/OFF) (got '${value}')")
  endif()
endfunction()

# _therock_resolve_debug_info_settings
# Resolves the effective debug-info settings for a subproject, or the global
# baseline when target_name is empty. Results are returned via the out_*
# variables (ON/OFF for generate/split).
#
# Precedence for THEROCK_GENERATE_DEBUG_INFO:
#   1. <target_name>_GENERATE_DEBUG_INFO  (per-subproject explicit override)
#   2. THEROCK_GENERATE_DEBUG_INFO if explicitly set (tri-state: non-empty)
#   3. build-type default derived from the subproject's effective build type
#      (<target_name>_BUILD_TYPE else CMAKE_BUILD_TYPE): ON for Debug or
#      RelWithDebInfo, otherwise OFF.
# THEROCK_DEBUG_INFO_LEVEL: <target_name>_DEBUG_INFO_LEVEL else the global value.
# THEROCK_SPLIT_DEBUG_INFO: always ON on Windows; otherwise
#   <target_name>_SPLIT_DEBUG_INFO else the global value.
#
# Per-subproject values are validated here (the global values are validated
# where they are declared in the top-level CMakeLists.txt).
function(_therock_resolve_debug_info_settings target_name out_generate out_level out_split)
  # Effective build type (mirrors the CMAKE_BUILD_TYPE derivation used below).
  set(_effective_build_type "${${target_name}_BUILD_TYPE}")
  if(NOT _effective_build_type)
    set(_effective_build_type "${CMAKE_BUILD_TYPE}")
  endif()
  if(_effective_build_type STREQUAL "Debug" OR _effective_build_type STREQUAL "RelWithDebInfo")
    set(_build_type_generate ON)
  else()
    set(_build_type_generate OFF)
  endif()

  # Resolve GENERATE_DEBUG_INFO.
  if(target_name AND DEFINED ${target_name}_GENERATE_DEBUG_INFO)
    _therock_normalize_bool(_generate "${${target_name}_GENERATE_DEBUG_INFO}"
      "${target_name}_GENERATE_DEBUG_INFO")
  elseif(NOT "${THEROCK_GENERATE_DEBUG_INFO}" STREQUAL "")
    # Global flag explicitly set (already canonicalized to ON/OFF at declaration).
    set(_generate "${THEROCK_GENERATE_DEBUG_INFO}")
  else()
    set(_generate "${_build_type_generate}")
  endif()

  # Resolve DEBUG_INFO_LEVEL.
  if(target_name AND DEFINED ${target_name}_DEBUG_INFO_LEVEL)
    set(_level "${${target_name}_DEBUG_INFO_LEVEL}")
    if(NOT _level MATCHES "^(minimal|full|extra)$")
      message(FATAL_ERROR
        "${target_name}_DEBUG_INFO_LEVEL must be 'minimal', 'full' or 'extra' (got '${_level}')")
    endif()
    if(WIN32 AND NOT _level MATCHES "^(minimal|full)$")
      message(WARNING
        "${target_name}_DEBUG_INFO_LEVEL='${_level}' is not supported on Windows; "
        "only 'minimal' or 'full' apply (treated as 'full').")
    endif()
  else()
    set(_level "${THEROCK_DEBUG_INFO_LEVEL}")
  endif()

  # Resolve SPLIT_DEBUG_INFO.
  if(WIN32)
    if(target_name AND DEFINED ${target_name}_SPLIT_DEBUG_INFO AND NOT ${target_name}_SPLIT_DEBUG_INFO)
      message(WARNING
        "${target_name}_SPLIT_DEBUG_INFO=OFF is not supported on Windows, as Windows does "
        "not support embedded debug information. Overriding to ON.")
    endif()
    set(_split ON)
  elseif(target_name AND DEFINED ${target_name}_SPLIT_DEBUG_INFO)
    _therock_normalize_bool(_split "${${target_name}_SPLIT_DEBUG_INFO}"
      "${target_name}_SPLIT_DEBUG_INFO")
  elseif(THEROCK_SPLIT_DEBUG_INFO)
    set(_split ON)
  else()
    set(_split OFF)
  endif()

  # Warning if generation disabled + explicit level/split set. User probably forgot to turn on generation.
  if(target_name AND NOT _generate)
    if(DEFINED ${target_name}_DEBUG_INFO_LEVEL)
      message(WARNING
        "${target_name}_DEBUG_INFO_LEVEL is set but debug info generation is disabled for "
        "'${target_name}'; the value is ignored.")
    endif()
    if(NOT WIN32 AND DEFINED ${target_name}_SPLIT_DEBUG_INFO)
      message(WARNING
        "${target_name}_SPLIT_DEBUG_INFO is set but debug info generation is disabled for "
        "'${target_name}'; the value is ignored.")
    endif()
  endif()

  set(${out_generate} "${_generate}" PARENT_SCOPE)
  set(${out_level} "${_level}" PARENT_SCOPE)
  set(${out_split} "${_split}" PARENT_SCOPE)
endfunction()
