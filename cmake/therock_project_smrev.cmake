# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

option(THEROCK_PROJECT_SMREV
  "Use reviewed project source identities for Git sources (local Linux experiment)" OFF)

# Reusable lists only: each target explicitly selects the sets it consumes.
# They are not implicit inputs for every project.
set(THEROCK_PROJECT_ROOT_INPUTS
  CMakeLists.txt VERSION VERSION.txt cmake cmake_modules version.json)
set(THEROCK_PROJECT_ROCM_VERSION_ENVIRONMENT
  ROCM_BUILD_ID ROCM_LIBPATCH_VERSION SOURCE_DATE_EPOCH)
set(THEROCK_PROJECT_PACKAGE_ENVIRONMENT
  CPACK_DEBIAN_PACKAGE_RELEASE CPACK_RPM_PACKAGE_RELEASE)
set(THEROCK_PROJECT_LIBRARY_ENVIRONMENT
  CC CXX DESTDIR LLVM_TOOLS_SEARCH_PREFIX ROCM_LIBS_ENABLE_ROOT_CTEST ROCM_PATH)

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
  set(_generator "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../build_tools/generate_project_smrev.py")
  get_target_property(_declared "${target_name}" THEROCK_FPRINT_PROJECT_SOURCE)
  if(NOT _declared)
    # A newly introduced monorepo target must not silently use repository HEAD.
    foreach(_root IN ITEMS "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}" "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}")
      if(_root)
        file(REAL_PATH "${_root}" _repo)
        file(REAL_PATH "${source_dir}" _source)
        cmake_path(IS_PREFIX _repo "${_source}" NORMALIZE _inside)
        if(_inside)
          message(FATAL_ERROR "Missing project source-identity declaration for ${target_name}")
        endif()
      endif()
    endforeach()
    return()
  endif()
  if(NOT target_name MATCHES "^[A-Za-z0-9_.+-]+$")
    message(FATAL_ERROR "Invalid source-identity target name: ${target_name}")
  endif()
  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
  find_package(Git REQUIRED)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --show-toplevel
    OUTPUT_VARIABLE _repo OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  file(REAL_PATH "${_repo}" _repo)
  # Use the configured checkout roots, including supported source overrides.
  # Logical repository names are stable across checkout locations.
  set(_repository)
  foreach(_name IN ITEMS rocm-systems rocm-libraries)
    string(REPLACE "-" "_" _variable "${_name}")
    string(TOUPPER "${_variable}" _variable)
    set(_configured_root "${THEROCK_${_variable}_SOURCE_DIR}")
    if(_configured_root)
      file(REAL_PATH "${_configured_root}" _configured_root)
      if(_repo STREQUAL _configured_root)
        set(_repository "${_name}")
      endif()
    endif()
  endforeach()
  if(NOT _repository)
    file(REAL_PATH "${THEROCK_SOURCE_DIR}" _therock_root)
    cmake_path(IS_PREFIX _therock_root "${_repo}" NORMALIZE _inside)
    if(NOT _inside OR _repo STREQUAL _therock_root)
      message(FATAL_ERROR "${target_name}: cannot identify a separate source repository at ${_repo}")
    endif()
    cmake_path(RELATIVE_PATH _repo BASE_DIRECTORY "${_therock_root}" OUTPUT_VARIABLE _repository)
  endif()
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
  set(_arguments --target "${target_name}" --source "${source_dir}"
    --repository "${_repository}" --output "${_marker}" --request "${_request}")
  get_target_property(_scope "${target_name}" THEROCK_FPRINT_PROJECT_SCOPE)
  if(_scope)
    list(APPEND _arguments "--scope=${_scope}")
  endif()
  foreach(_property IN ITEMS INPUTS OPTIONAL_INPUTS ENVIRONMENT ENVIRONMENT_PATTERNS VERSION_QUERIES)
    if(_property STREQUAL "INPUTS")
      set(_argument input)
    elseif(_property STREQUAL "OPTIONAL_INPUTS")
      set(_argument optional-input)
    elseif(_property STREQUAL "ENVIRONMENT")
      set(_argument environment)
    elseif(_property STREQUAL "ENVIRONMENT_PATTERNS")
      set(_argument environment-pattern)
    else()
      set(_argument version-query)
    endif()
    get_target_property(_values "${target_name}" "THEROCK_FPRINT_PROJECT_${_property}")
    if(_values)
      foreach(_value IN LISTS _values)
        list(APPEND _arguments "--${_argument}=${_value}")
      endforeach()
    endif()
  endforeach()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_generator}" ${_arguments}
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
  # CMake already tracks the declaration files it evaluates. Track the Python
  # generator too; the request file captures the resolved configured inputs.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_generator}")
  # HEAD/ref changes cause an automatic reconfigure. Working-tree and environment
  # changes are also checked by the build guard, even if CMake was not rerun.
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --absolute-git-dir
    OUTPUT_VARIABLE _git_dir OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --path-format=absolute --git-common-dir
    OUTPUT_VARIABLE _common_dir OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  file(GLOB_RECURSE _refs CONFIGURE_DEPENDS "${_common_dir}/refs/*")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_git_dir}/HEAD" ${_refs})
  if(EXISTS "${_common_dir}/packed-refs")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_common_dir}/packed-refs")
  endif()
  get_target_property(_reason "${target_name}" THEROCK_FPRINT_PROJECT_UNCACHEABLE_REASON)
  if(_reason)
    message(STATUS "${target_name}: source marker generated, artifact fingerprint disabled: ${_reason}")
    set(${out_cacheable} FALSE PARENT_SCOPE)
  endif()
  set(${out_marker} "${_marker}" PARENT_SCOPE)
endfunction()
