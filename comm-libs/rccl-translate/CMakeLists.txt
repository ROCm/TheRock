# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(rccl-translate LANGUAGES NONE)

find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)

option(
  RCCL_TRANSLATE_FAIL_ON_SKIPPED
  "Fail when RocJitsu rejects a discovered RCCL code object"
  OFF
)

foreach(_required_path
    RCCL_ROOT
    ROCJITSU_ROOT
    ROCJITSU_HOTSWAP_ROOT
    KPACK_PYTHON_SOURCE
)
  if(NOT IS_ABSOLUTE "${${_required_path}}")
    message(FATAL_ERROR "${_required_path} must be an absolute path")
  endif()
endforeach()

find_library(RCCL_LIBRARY
  NAMES rccl
  PATHS "${RCCL_ROOT}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
find_program(ROCJITSU_PRETRANSLATE_TOOL
  NAMES rj_pretranslate
  PATHS "${ROCJITSU_ROOT}/bin"
  NO_DEFAULT_PATH
  REQUIRED
)
find_program(ROCJITSU_PRETRANSLATE_DRIVER
  NAMES rocjitsu-pretranslate.py
  PATHS "${ROCJITSU_ROOT}/bin"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(ROCJITSU_HOTSWAP_TRANSLATOR_LIBRARY
  NAMES rocjitsu_gfx1250_b0_to_a0
  PATHS "${ROCJITSU_HOTSWAP_ROOT}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)

set(_unbundled_dir "${CMAKE_CURRENT_BINARY_DIR}/unbundled")
set(_cache_root "${CMAKE_CURRENT_BINARY_DIR}/cache")
set(_report_dir "${CMAKE_CURRENT_BINARY_DIR}/report")
set(_temporary_root "${CMAKE_CURRENT_BINARY_DIR}/temporary")
set(_translation_stamp "${CMAKE_CURRENT_BINARY_DIR}/translation.stamp")
set(_strict_args)
if(RCCL_TRANSLATE_FAIL_ON_SKIPPED)
  list(APPEND _strict_args --fail-on-skipped)
endif()

add_custom_command(
  OUTPUT "${_translation_stamp}"
  COMMENT "Pre-translating RCCL gfx1250 device code"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf
    "${_unbundled_dir}" "${_cache_root}" "${_report_dir}" "${_temporary_root}"
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${KPACK_PYTHON_SOURCE}"
    "${Python3_EXECUTABLE}" -m rocm_kpack.tools.bulk_unbundle
    --gfx-arch gfx1250
    --output-dir "${_unbundled_dir}"
    "${RCCL_LIBRARY}"
  COMMAND "${CMAKE_COMMAND}" -E env
    "LD_LIBRARY_PATH=${ROCJITSU_HOTSWAP_ROOT}/lib:${ROCJITSU_ROOT}/lib"
    "${Python3_EXECUTABLE}" "${ROCJITSU_PRETRANSLATE_DRIVER}"
    --tool "${ROCJITSU_PRETRANSLATE_TOOL}"
    --store-root "${_cache_root}"
    --report-dir "${_report_dir}"
    --temporary-root "${_temporary_root}"
    --target gfx1250
    --skip-kpack
    --skip-compressed
    --portable
    ${_strict_args}
    "${_unbundled_dir}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${_translation_stamp}"
  DEPENDS
    "${RCCL_LIBRARY}"
    "${ROCJITSU_PRETRANSLATE_TOOL}"
    "${ROCJITSU_PRETRANSLATE_DRIVER}"
    "${ROCJITSU_HOTSWAP_TRANSLATOR_LIBRARY}"
    "${KPACK_PYTHON_SOURCE}/rocm_kpack/binutils.py"
    "${KPACK_PYTHON_SOURCE}/rocm_kpack/ccob_parser.py"
    "${KPACK_PYTHON_SOURCE}/rocm_kpack/tools/bulk_unbundle.py"
  VERBATIM
)

add_custom_target(rccl-translate-cache ALL DEPENDS "${_translation_stamp}")

install(
  DIRECTORY "${_cache_root}/"
  DESTINATION share/rocjitsu/translations
)
