# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

option(THEROCK_PROJECT_SMREV
  "Use reviewed project source identities for Git sources (local Linux experiment)" OFF)

# Return a freshly validated marker. The regular sibling-marker/Git reader is
# retained for all other routes. It must not revalidate this marker against the
# containing TheRock repository: build/ is often inside that unrelated checkout.
function(therock_project_smrev out_marker out_cacheable target_name source_dir)
  set(${out_marker} "" PARENT_SCOPE)
  set(${out_cacheable} TRUE PARENT_SCOPE)
  if(NOT THEROCK_PROJECT_SMREV)
    return()
  endif()
  if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" OR WIN32)
    message(FATAL_ERROR "THEROCK_PROJECT_SMREV currently supports native Linux builds")
  endif()
  set(_catalog "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../build_tools/project_smrev_inputs.json")
  set(_generator "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../build_tools/generate_project_smrev.py")
  file(READ "${_catalog}" _policies)
  string(JSON _policy ERROR_VARIABLE _missing GET "${_policies}" targets "${target_name}")
  if(_missing)
    # A newly introduced monorepo target must not silently use repository HEAD.
    foreach(_root IN ITEMS "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}" "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}")
      if(_root)
        file(REAL_PATH "${_root}" _repo)
        file(REAL_PATH "${source_dir}" _source)
        cmake_path(IS_PREFIX _repo "${_source}" NORMALIZE _inside)
        if(_inside)
          message(FATAL_ERROR "Missing project source-identity policy for ${target_name}")
        endif()
      endif()
    endforeach()
    return()
  endif()
  if(NOT target_name MATCHES "^[A-Za-z0-9_.+-]+$")
    message(FATAL_ERROR "Invalid source-identity target name: ${target_name}")
  endif()
  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
  set(_directory "${CMAKE_BINARY_DIR}/source-identities")
  set(_marker "${_directory}/.${target_name}.smrev")
  set(_request "${_directory}/${target_name}.request.json")
  set(_list "${_directory}/requests.txt")
  get_property(_initialized GLOBAL PROPERTY THEROCK_PROJECT_SMREV_INITIALIZED)
  if(NOT _initialized)
    file(MAKE_DIRECTORY "${_directory}")
    file(WRITE "${_list}" "")
    add_custom_target(therock-check-project-identities
      COMMAND "${Python3_EXECUTABLE}" "${_generator}" --check-list "${_list}"
      COMMENT "Checking configured project source identities"
      VERBATIM)
    set_property(GLOBAL PROPERTY THEROCK_PROJECT_SMREV_INITIALIZED TRUE)
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_generator}"
      --target "${target_name}" --source "${source_dir}"
      --output "${_marker}" --request "${_request}"
    RESULT_VARIABLE _result OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
  if(NOT _result STREQUAL "0")
    message(FATAL_ERROR "${target_name} source identity failed; no Git fallback is allowed:\n${_stderr}")
  endif()
  file(APPEND "${_list}" "${_request}\n")
  # Publish the effective identity directory without changing the real source.
  file(MAKE_DIRECTORY "${_directory}/${target_name}")
  set_property(TARGET "${target_name}" PROPERTY THEROCK_FPRINT_SOURCE_DIR "${_directory}/${target_name}")
  set_property(TARGET "${target_name}" PROPERTY THEROCK_PROJECT_SMREV_CHECK_COMMAND
    "${Python3_EXECUTABLE}" "${_generator}" --check-request "${_request}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_catalog}" "${_generator}")
  # HEAD/ref changes cause an automatic reconfigure. Working-tree and environment
  # changes are also checked by the build guard, even if CMake was not rerun.
  find_package(Git REQUIRED)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --absolute-git-dir
    OUTPUT_VARIABLE _git_dir OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --path-format=absolute --git-common-dir
    OUTPUT_VARIABLE _common_dir OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  file(GLOB_RECURSE _refs CONFIGURE_DEPENDS "${_common_dir}/refs/*")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_git_dir}/HEAD" ${_refs})
  if(EXISTS "${_common_dir}/packed-refs")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_common_dir}/packed-refs")
  endif()
  string(JSON _reason ERROR_VARIABLE _no_reason GET "${_policy}" uncacheable_reason)
  if(NOT _no_reason)
    message(STATUS "${target_name}: source marker generated, artifact fingerprint disabled: ${_reason}")
    set(${out_cacheable} FALSE PARENT_SCOPE)
  endif()
  set(${out_marker} "${_marker}" PARENT_SCOPE)
endfunction()
