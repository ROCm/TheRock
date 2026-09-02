# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Builds the standalone rocm-systems/python/rocshmem PEP-517 project.

cmake_minimum_required(VERSION 3.25)

project(therock-rocshmem4py NONE)

set(ROCSHMEM4PY_SOURCE_DIR "" CACHE PATH
  "Source directory of the standalone rocshmem4py project")
if(NOT ROCSHMEM4PY_SOURCE_DIR)
  message(FATAL_ERROR
    "ROCSHMEM4PY_SOURCE_DIR must point at the standalone rocshmem4py project")
endif()
if(NOT EXISTS "${ROCSHMEM4PY_SOURCE_DIR}/pyproject.toml")
  message(FATAL_ERROR
    "No pyproject.toml under ROCSHMEM4PY_SOURCE_DIR "
    "('${ROCSHMEM4PY_SOURCE_DIR}'): the rocm-systems checkout does not contain "
    "the standalone rocshmem4py project.")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(ROCSHMEM4PY_PYTHON_EXECUTABLES "" CACHE STRING
  "Python interpreters for which rocshmem4py wheels are built")
if(NOT ROCSHMEM4PY_PYTHON_EXECUTABLES)
  set(ROCSHMEM4PY_PYTHON_EXECUTABLES "${Python3_EXECUTABLE}")
endif()

set(ROCSHMEM4PY_FINALIZE_SCRIPT "" CACHE FILEPATH
  "TheRock script that finalizes and validates built wheels")
if(NOT EXISTS "${ROCSHMEM4PY_FINALIZE_SCRIPT}")
  message(FATAL_ERROR
    "ROCSHMEM4PY_FINALIZE_SCRIPT does not exist: '${ROCSHMEM4PY_FINALIZE_SCRIPT}'")
endif()

if(NOT THEROCK_PACKAGE_VERSION)
  message(FATAL_ERROR
    "THEROCK_PACKAGE_VERSION is required to version and runtime-pin rocshmem4py")
endif()

find_package(ROCSHMEM 3.5 CONFIG REQUIRED)
get_filename_component(_rocshmem_prefix "${rocshmem_INCLUDE_DIRS}/.." ABSOLUTE)
set(_build_prefix_path "${_rocshmem_prefix};${CMAKE_PREFIX_PATH}")
string(REPLACE ";" ":" _build_prefix_path "${_build_prefix_path}")

find_program(ROCSHMEM4PY_PATCHELF NAMES patchelf REQUIRED)

set(_raw_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/raw-wheelhouse")
set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/wheelhouse")
set(_wheel_stamp "${CMAKE_CURRENT_BINARY_DIR}/wheel.stamp")
set(_build_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")

# Build from a private copy so PEP-517 state never pollutes the source checkout.
file(GLOB_RECURSE _rocshmem4py_source_files
  CONFIGURE_DEPENDS
  LIST_DIRECTORIES FALSE
  "${ROCSHMEM4PY_SOURCE_DIR}/*"
)
list(FILTER _rocshmem4py_source_files EXCLUDE REGEX
  "/(build|dist|[^/]+\\.egg-info)/"
)

set(_wheel_build_commands)
foreach(_python IN LISTS ROCSHMEM4PY_PYTHON_EXECUTABLES)
  if(NOT EXISTS "${_python}")
    message(FATAL_ERROR "rocshmem4py Python interpreter does not exist: ${_python}")
  endif()
  list(APPEND _wheel_build_commands
    COMMAND "${CMAKE_COMMAND}" -E env
      "CMAKE_PREFIX_PATH=${_build_prefix_path}"
      "THEROCK_TOOLCHAIN_ROOT=${THEROCK_TOOLCHAIN_ROOT}"
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
    "${ROCSHMEM4PY_SOURCE_DIR}" "${_build_source_dir}"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf
    "${_build_source_dir}/build"
    "${_build_source_dir}/dist"
    "${_build_source_dir}/rocshmem4py.egg-info"
  ${_wheel_build_commands}
  COMMAND "${Python3_EXECUTABLE}" "${ROCSHMEM4PY_FINALIZE_SCRIPT}"
    --input-dir "${_raw_wheel_dir}"
    --output-dir "${_wheel_dir}"
    --rocm-version "${THEROCK_PACKAGE_VERSION}"
    --patchelf "${ROCSHMEM4PY_PATCHELF}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${_wheel_stamp}"
  DEPENDS
    ${_rocshmem4py_source_files}
    "${ROCSHMEM4PY_FINALIZE_SCRIPT}"
  COMMENT "Building and validating rocshmem4py wheels for ${ROCSHMEM4PY_PYTHON_EXECUTABLES}"
  VERBATIM
)

add_custom_target(rocshmem4py-wheel ALL DEPENDS "${_wheel_stamp}")

install(
  DIRECTORY "${_wheel_dir}/"
  DESTINATION "share/rocshmem4py/wheels"
  FILES_MATCHING PATTERN "*.whl"
)
