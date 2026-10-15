# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_python_wheel_subproject
# Declares, activates, and provides an artifact for a sub-project that builds
# an external PEP-517 Python project with compiled extensions into wheels, one
# per interpreter from therock_detect_python_versions(). Wheels are installed
# to share/<name>/wheels and published by build_tools/build_python_packages.py.
#
# Projects should take TheRock's package version, rocm-sdk-core requirement,
# and extension RPATH as build inputs so their wheels need no post-build
# changes.
#
# Arguments:
# NAME: Sub-project and artifact name.
# SOURCE_DIR: Absolute path to the PEP-517 project (containing pyproject.toml).
# FPRINT_SOURCE_DIR: Source fingerprint directory. Defaults to SOURCE_DIR.
# FINALIZE_SCRIPT: Optional compatibility bridge for projects that cannot yet
#   take those build inputs. Invoked as
#   `<script> --input-dir <raw wheels> --output-dir <wheels>
#    --rocm-version <THEROCK_PACKAGE_VERSION> --patchelf <patchelf>`.
# ARTIFACT_DESCRIPTOR: Artifact descriptor, relative to the calling directory.
#   It must include share/<name>/wheels/*.whl in the lib component.
# ENV_FROM_VARS: Names of sub-project CMake variables (i.e. those set by the
#   compiler toolchain, such as THEROCK_TOOLCHAIN_ROOT) to export into the
#   wheel build environment.
# COMPILER_TOOLCHAIN: As for therock_cmake_subproject_declare.
# BUILD_DEPS: As for therock_cmake_subproject_declare. Their dist directories
#   are prepended to CMAKE_PREFIX_PATH for the wheel build.
# RUNTIME_DEPS: As for therock_cmake_subproject_declare.
function(therock_python_wheel_subproject target_name)
  cmake_parse_arguments(
    PARSE_ARGV 1 ARG
    ""
    "SOURCE_DIR;FPRINT_SOURCE_DIR;FINALIZE_SCRIPT;ARTIFACT_DESCRIPTOR;COMPILER_TOOLCHAIN"
    "ENV_FROM_VARS;BUILD_DEPS;RUNTIME_DEPS"
  )
  foreach(_required SOURCE_DIR ARTIFACT_DESCRIPTOR)
    if(NOT ARG_${_required})
      message(FATAL_ERROR "therock_python_wheel_subproject(${target_name}): ${_required} is required")
    endif()
  endforeach()
  if(NOT ARG_FPRINT_SOURCE_DIR)
    set(ARG_FPRINT_SOURCE_DIR "${ARG_SOURCE_DIR}")
  endif()

  therock_detect_python_versions(_python_executables _python_versions)

  set(_prefix_path)
  foreach(_dep IN LISTS ARG_BUILD_DEPS)
    get_target_property(_dep_dist_dir "${_dep}" THEROCK_DIST_DIR)
    if(NOT _dep_dist_dir)
      message(FATAL_ERROR "therock_python_wheel_subproject(${target_name}): BUILD_DEPS entry '${_dep}' is not a declared sub-project")
    endif()
    list(APPEND _prefix_path "${_dep_dist_dir}")
  endforeach()

  # CMAKE_ARGS is stored in a target property and expanded again by the
  # subproject's configure command. Escape the list separators across those
  # boundaries so the child receives each list as a single -D value.
  list(JOIN _python_executables "\;" _python_executables_str)
  list(JOIN _prefix_path "\;" _prefix_path_str)
  list(JOIN ARG_ENV_FROM_VARS "\;" _env_from_vars_str)

  set(_adapter_dir "${THEROCK_SOURCE_DIR}/cmake/therock_python_wheel")
  set(_fprint_globs "${_adapter_dir}/*")
  if(ARG_FINALIZE_SCRIPT)
    list(APPEND _fprint_globs
      "${ARG_FINALIZE_SCRIPT}"
      "${THEROCK_SOURCE_DIR}/build_tools/third_party/change_wheel_version/change_wheel_version.py"
    )
  endif()

  therock_cmake_subproject_declare(${target_name}
    EXTERNAL_SOURCE_DIR "${_adapter_dir}"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${target_name}"
    FPRINT_SOURCE_DIR "${ARG_FPRINT_SOURCE_DIR}"
    FPRINT_FILE_GLOBS
      ${_fprint_globs}
    DISABLE_AMDGPU_TARGETS
    BACKGROUND_BUILD
    CMAKE_ARGS
      "-DTHEROCK_PYTHON_WHEEL_NAME=${target_name}"
      "-DTHEROCK_PYTHON_WHEEL_SOURCE_DIR=${ARG_SOURCE_DIR}"
      "-DTHEROCK_PYTHON_WHEEL_PYTHON_EXECUTABLES:STRING=${_python_executables_str}"
      "-DTHEROCK_PYTHON_WHEEL_PREFIX_PATH:STRING=${_prefix_path_str}"
      "-DTHEROCK_PYTHON_WHEEL_ENV_FROM_VARS:STRING=${_env_from_vars_str}"
      "-DTHEROCK_PYTHON_WHEEL_FINALIZE_SCRIPT=${ARG_FINALIZE_SCRIPT}"
      "-DTHEROCK_PACKAGE_VERSION=${THEROCK_PACKAGE_VERSION}"
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    COMPILER_TOOLCHAIN
      ${ARG_COMPILER_TOOLCHAIN}
    BUILD_DEPS
      ${ARG_BUILD_DEPS}
    RUNTIME_DEPS
      ${ARG_RUNTIME_DEPS}
  )
  therock_cmake_subproject_activate(${target_name})

  therock_provide_artifact(${target_name}
    TARGET_NEUTRAL
    DESCRIPTOR "${ARG_ARTIFACT_DESCRIPTOR}"
    COMPONENTS
      lib
    SUBPROJECT_DEPS
      ${target_name}
  )
endfunction()
