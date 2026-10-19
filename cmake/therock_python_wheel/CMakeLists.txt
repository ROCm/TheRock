# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Builds an external PEP-517 project into wheels as a TheRock
# sub-project. Declared via therock_python_wheel_subproject(); see
# cmake/therock_python_wheel.cmake for the meaning of each input.

cmake_minimum_required(VERSION 3.25)

project(therock-python-wheel NONE)

set(THEROCK_PYTHON_WHEEL_NAME "" CACHE STRING
  "Name of the wheel sub-project; wheels install to share/<name>/wheels")
if(NOT THEROCK_PYTHON_WHEEL_NAME)
  message(FATAL_ERROR "THEROCK_PYTHON_WHEEL_NAME is required")
endif()

set(THEROCK_PYTHON_WHEEL_SOURCE_DIR "" CACHE PATH
  "Source directory of the PEP-517 project")
if(NOT THEROCK_PYTHON_WHEEL_SOURCE_DIR)
  message(FATAL_ERROR
    "THEROCK_PYTHON_WHEEL_SOURCE_DIR must point at a PEP-517 project")
endif()
if(NOT EXISTS "${THEROCK_PYTHON_WHEEL_SOURCE_DIR}/pyproject.toml")
  message(FATAL_ERROR
    "No pyproject.toml under THEROCK_PYTHON_WHEEL_SOURCE_DIR "
    "('${THEROCK_PYTHON_WHEEL_SOURCE_DIR}'): the source checkout does not "
    "contain the ${THEROCK_PYTHON_WHEEL_NAME} project.")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(THEROCK_PYTHON_WHEEL_PYTHON_EXECUTABLES "" CACHE STRING
  "Python interpreters for which wheels are built")
if(NOT THEROCK_PYTHON_WHEEL_PYTHON_EXECUTABLES)
  set(THEROCK_PYTHON_WHEEL_PYTHON_EXECUTABLES "${Python3_EXECUTABLE}")
endif()

set(THEROCK_PYTHON_WHEEL_PREFIX_PATH "" CACHE STRING
  "Install prefixes of build dependencies, searched before CMAKE_PREFIX_PATH")
set(THEROCK_PYTHON_WHEEL_ENV_FROM_VARS "" CACHE STRING
  "Names of CMake variables exported into the wheel build environment")

set(THEROCK_PYTHON_WHEEL_FINALIZE_SCRIPT "" CACHE FILEPATH
  "Optional compatibility script that finalizes built wheels")

set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/wheelhouse")
set(_wheel_stamp "${CMAKE_CURRENT_BINARY_DIR}/wheel.stamp")
set(_build_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")

set(_finalize_commands)
if(THEROCK_PYTHON_WHEEL_FINALIZE_SCRIPT)
  if(NOT EXISTS "${THEROCK_PYTHON_WHEEL_FINALIZE_SCRIPT}")
    message(FATAL_ERROR
      "THEROCK_PYTHON_WHEEL_FINALIZE_SCRIPT does not exist: "
      "'${THEROCK_PYTHON_WHEEL_FINALIZE_SCRIPT}'")
  endif()
  if(NOT THEROCK_PACKAGE_VERSION)
    message(FATAL_ERROR
      "THEROCK_PACKAGE_VERSION is required to version and runtime-pin "
      "${THEROCK_PYTHON_WHEEL_NAME}")
  endif()
  find_program(THEROCK_PYTHON_WHEEL_PATCHELF NAMES patchelf REQUIRED)
  set(_raw_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/raw-wheelhouse")
  set(_finalize_commands
    COMMAND "${Python3_EXECUTABLE}" "${THEROCK_PYTHON_WHEEL_FINALIZE_SCRIPT}"
      --input-dir "${_raw_wheel_dir}"
      --output-dir "${_wheel_dir}"
      --rocm-version "${THEROCK_PACKAGE_VERSION}"
      --patchelf "${THEROCK_PYTHON_WHEEL_PATCHELF}"
  )
else()
  set(_raw_wheel_dir "${_wheel_dir}")
endif()

set(_build_prefix_path ${THEROCK_PYTHON_WHEEL_PREFIX_PATH} ${CMAKE_PREFIX_PATH})
string(REPLACE ";" ":" _build_prefix_path "${_build_prefix_path}")

set(_build_env "CMAKE_PREFIX_PATH=${_build_prefix_path}")
foreach(_var IN LISTS THEROCK_PYTHON_WHEEL_ENV_FROM_VARS)
  list(APPEND _build_env "${_var}=${${_var}}")
endforeach()

# Build from a private copy so PEP-517 state never pollutes the source checkout.
file(GLOB_RECURSE _source_files
  CONFIGURE_DEPENDS
  LIST_DIRECTORIES FALSE
  "${THEROCK_PYTHON_WHEEL_SOURCE_DIR}/*"
)
list(FILTER _source_files EXCLUDE REGEX
  "/(build|dist|[^/]+\\.egg-info)/"
)

set(_wheel_build_commands)
foreach(_python IN LISTS THEROCK_PYTHON_WHEEL_PYTHON_EXECUTABLES)
  if(NOT EXISTS "${_python}")
    message(FATAL_ERROR "Python interpreter does not exist: ${_python}")
  endif()
  list(APPEND _wheel_build_commands
    COMMAND "${CMAKE_COMMAND}" -E env ${_build_env}
      --
      "${_python}" -m pip wheel
        --disable-pip-version-check
        --no-deps
        --wheel-dir "${_raw_wheel_dir}"
        "${_build_source_dir}"
  )
endforeach()

add_custom_command(
  OUTPUT "${_wheel_stamp}"
  # Do not reuse PEP-517 state or raw wheels from an earlier build.
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_raw_wheel_dir}"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_build_source_dir}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
    "${THEROCK_PYTHON_WHEEL_SOURCE_DIR}" "${_build_source_dir}"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf
    "${_build_source_dir}/build"
    "${_build_source_dir}/dist"
    "${_build_source_dir}/${THEROCK_PYTHON_WHEEL_NAME}.egg-info"
  ${_wheel_build_commands}
  ${_finalize_commands}
  COMMAND "${CMAKE_COMMAND}" -E touch "${_wheel_stamp}"
  DEPENDS
    ${_source_files}
    ${THEROCK_PYTHON_WHEEL_FINALIZE_SCRIPT}
  COMMENT "Building ${THEROCK_PYTHON_WHEEL_NAME} wheels for ${THEROCK_PYTHON_WHEEL_PYTHON_EXECUTABLES}"
  VERBATIM
)

add_custom_target(python-wheel ALL DEPENDS "${_wheel_stamp}")

install(
  DIRECTORY "${_wheel_dir}/"
  DESTINATION "share/${THEROCK_PYTHON_WHEEL_NAME}/wheels"
  FILES_MATCHING PATTERN "*.whl"
)
