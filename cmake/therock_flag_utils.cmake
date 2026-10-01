# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_flag_utils.cmake
# Centralized flag system for TheRock build infrastructure.
#
# Flags are build-system-level controls that affect how subprojects are
# configured. Unlike features (therock_features.cmake) which control
# subproject inclusion, flags control variable propagation and compiler
# defines within included subprojects.
#
# Each flag results in the following changes:
#   THEROCK_FLAG_${NAME}: typed cache variable controlling the flag state
#   Optionally propagates CMAKE variables and CPP defines to all or specific
#   subprojects via the project_init.cmake mechanism.
#
# See docs/development/flags.md for full documentation.

# Global property to track all declared flags.
set_property(GLOBAL PROPERTY THEROCK_ALL_FLAGS)

set(_THEROCK_BUILD_FLAGS_PROTOCOL_VERSION 1)

function(_therock_normalize_bool out_var value context)
  string(TOUPPER "${value}" _value_upper)
  if(_value_upper STREQUAL "1" OR
      _value_upper STREQUAL "ON" OR
      _value_upper STREQUAL "YES" OR
      _value_upper STREQUAL "TRUE" OR
      _value_upper STREQUAL "Y")
    set(${out_var} "1" PARENT_SCOPE)
  elseif(_value_upper STREQUAL "0" OR
      _value_upper STREQUAL "OFF" OR
      _value_upper STREQUAL "NO" OR
      _value_upper STREQUAL "FALSE" OR
      _value_upper STREQUAL "N")
    set(${out_var} "0" PARENT_SCOPE)
  else()
    message(FATAL_ERROR
      "${context}: BOOL value '${value}' is invalid; use ON/OFF, TRUE/FALSE, "
      "YES/NO, Y/N, or 1/0"
    )
  endif()
endfunction()

function(_therock_normalize_integer out_var value context)
  if(NOT "${value}" MATCHES "^(0|-?[1-9][0-9]*)$")
    message(FATAL_ERROR
      "${context}: INTEGER value '${value}' is invalid; use canonical signed "
      "base-10 spelling without a leading plus sign or leading zeroes"
    )
  endif()
  set(${out_var} "${value}" PARENT_SCOPE)
endfunction()

function(_therock_normalize_flag_value out_var flag_name value context)
  get_property(_type GLOBAL PROPERTY _THEROCK_FLAG_${flag_name}_TYPE)
  if(_type STREQUAL "BOOL")
    _therock_normalize_bool(_normalized "${value}" "${context}")
  elseif(_type STREQUAL "INTEGER")
    _therock_normalize_integer(_normalized "${value}" "${context}")
    get_property(_valid_values GLOBAL PROPERTY _THEROCK_FLAG_${flag_name}_VALID_VALUES)
    if(NOT "${_valid_values}" STREQUAL "" AND
        NOT "${_normalized}" IN_LIST _valid_values)
      message(FATAL_ERROR
        "${context}: INTEGER value '${value}' is not one of the allowed values: "
        "${_valid_values}"
      )
    endif()
  else()
    message(FATAL_ERROR "${context}: unsupported flag type '${_type}'")
  endif()
  set(${out_var} "${_normalized}" PARENT_SCOPE)
endfunction()

# therock_declare_flag
# Declares a build flag with optional variable and define propagation.
#
# Arguments:
#   NAME           - Unique flag identifier (creates THEROCK_FLAG_${NAME} cache var)
#   TYPE           - BOOL (default) or INTEGER
#   DEFAULT_VALUE  - Typed default value
#   DESCRIPTION    - Short description for the cache variable
#   VALID_VALUES   - (Optional) Allowed values for INTEGER flags
#   ISSUE          - (Optional) Tracking issue URL
#   GLOBAL_PROPAGATE_FLAG
#                  - Propagate THEROCK_FLAG_${NAME} to all sub-projects
#                    regardless of whether the flag is enabled or disabled.
#   GLOBAL_CMAKE_VARS   - (Optional) VAR=VALUE pairs set in super-project and
#                          all sub-projects when the flag is enabled
#   GLOBAL_CPP_DEFINES  - (Optional) Preprocessor defines for all sub-projects
#                          when the flag is enabled
#   CMAKE_VARS          - (Optional) VAR=VALUE pairs set only in SUB_PROJECTS
#                          when the flag is enabled
#   CPP_DEFINES         - (Optional) Preprocessor defines only in SUB_PROJECTS
#                          when the flag is enabled
#   SUB_PROJECTS        - (Optional) List of sub-project target names for scoped
#                          CMAKE_VARS and CPP_DEFINES
function(therock_declare_flag)
  cmake_parse_arguments(PARSE_ARGV 0 ARG
    "GLOBAL_PROPAGATE_FLAG"
    "NAME;TYPE;DEFAULT_VALUE;DESCRIPTION;ISSUE"
    "VALID_VALUES;GLOBAL_CMAKE_VARS;GLOBAL_CPP_DEFINES;CMAKE_VARS;CPP_DEFINES;SUB_PROJECTS"
  )

  # Validate required arguments.
  if(NOT ARG_NAME)
    message(FATAL_ERROR "therock_declare_flag: NAME is required")
  endif()
  if(NOT DEFINED ARG_DEFAULT_VALUE)
    message(FATAL_ERROR "therock_declare_flag: DEFAULT_VALUE is required for flag ${ARG_NAME}")
  endif()
  if(NOT DEFINED ARG_DESCRIPTION OR "${ARG_DESCRIPTION}" STREQUAL "")
    message(FATAL_ERROR "therock_declare_flag: DESCRIPTION is required for flag ${ARG_NAME}")
  endif()
  if(NOT "${ARG_NAME}" MATCHES "^[A-Z][A-Z0-9_]*$")
    message(FATAL_ERROR
      "therock_declare_flag: NAME '${ARG_NAME}' must use uppercase letters, "
      "digits, and underscores, and must start with a letter"
    )
  endif()
  if(NOT ARG_TYPE)
    set(ARG_TYPE "BOOL")
  endif()
  if(NOT ARG_TYPE STREQUAL "BOOL" AND NOT ARG_TYPE STREQUAL "INTEGER")
    message(FATAL_ERROR
      "therock_declare_flag: TYPE for '${ARG_NAME}' must be BOOL or INTEGER"
    )
  endif()
  if(ARG_TYPE STREQUAL "BOOL" AND ARG_VALID_VALUES)
    message(FATAL_ERROR
      "therock_declare_flag: VALID_VALUES is only supported for INTEGER flags"
    )
  endif()
  if(ARG_TYPE STREQUAL "INTEGER" AND
      (ARG_GLOBAL_CMAKE_VARS OR ARG_GLOBAL_CPP_DEFINES OR
       ARG_CMAKE_VARS OR ARG_CPP_DEFINES))
    message(FATAL_ERROR
      "therock_declare_flag: enabled-only CMake variable and preprocessor "
      "propagation is only supported for BOOL flags; consume INTEGER flags "
      "through ROCMBuildFlags.cmake"
    )
  endif()

  # Check for duplicate flags.
  get_property(_all_flags GLOBAL PROPERTY THEROCK_ALL_FLAGS)
  if("${ARG_NAME}" IN_LIST _all_flags)
    message(FATAL_ERROR "therock_declare_flag: Flag '${ARG_NAME}' already declared")
  endif()

  # Validate that scoped vars/defines require SUB_PROJECTS.
  if((ARG_CMAKE_VARS OR ARG_CPP_DEFINES) AND NOT ARG_SUB_PROJECTS)
    message(FATAL_ERROR
      "therock_declare_flag: Flag '${ARG_NAME}' has CMAKE_VARS or CPP_DEFINES "
      "but no SUB_PROJECTS. Use GLOBAL_CMAKE_VARS/GLOBAL_CPP_DEFINES for "
      "project-wide settings, or specify SUB_PROJECTS for scoped settings."
    )
  endif()

  # Register the flag (metadata only — no cache/global manipulation here).
  # All cache variables and global state are created in therock_finalize_flags().
  set_property(GLOBAL APPEND PROPERTY THEROCK_ALL_FLAGS "${ARG_NAME}")

  # Store flag metadata in global properties for later retrieval.
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_TYPE "${ARG_TYPE}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_VALID_VALUES "${ARG_VALID_VALUES}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_DEFAULT_VALUE "${ARG_DEFAULT_VALUE}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_DESCRIPTION "${ARG_DESCRIPTION}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_GLOBAL_PROPAGATE_FLAG "${ARG_GLOBAL_PROPAGATE_FLAG}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_GLOBAL_CMAKE_VARS "${ARG_GLOBAL_CMAKE_VARS}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_GLOBAL_CPP_DEFINES "${ARG_GLOBAL_CPP_DEFINES}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_CMAKE_VARS "${ARG_CMAKE_VARS}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_CPP_DEFINES "${ARG_CPP_DEFINES}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_SUB_PROJECTS "${ARG_SUB_PROJECTS}")
  if(ARG_ISSUE)
    set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_ISSUE "${ARG_ISSUE}")
  endif()

  foreach(_valid_value ${ARG_VALID_VALUES})
    _therock_normalize_integer(
      _unused "${_valid_value}"
      "therock_declare_flag(${ARG_NAME}) VALID_VALUES"
    )
  endforeach()
  _therock_normalize_flag_value(
    _unused "${ARG_NAME}" "${ARG_DEFAULT_VALUE}"
    "therock_declare_flag(${ARG_NAME}) DEFAULT_VALUE"
  )
endfunction()

# therock_override_flag_default
# Changes the default value of a previously declared flag. Only updates the
# stored default property — actual cache variable creation happens in
# therock_finalize_flags(). Intended for use in BRANCH_FLAGS.cmake on
# integration branches.
function(therock_override_flag_default flag_name new_default)
  get_property(_all_flags GLOBAL PROPERTY THEROCK_ALL_FLAGS)
  if(NOT "${flag_name}" IN_LIST _all_flags)
    message(FATAL_ERROR
      "therock_override_flag_default: Flag '${flag_name}' has not been declared"
    )
  endif()

  message(STATUS "Flag ${flag_name} default overridden to ${new_default}")
  _therock_normalize_flag_value(
    _unused "${flag_name}" "${new_default}"
    "therock_override_flag_default(${flag_name})"
  )
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${flag_name}_DEFAULT_VALUE "${new_default}")
endfunction()

# therock_finalize_flags
# Processes all declared flags: sets global variables, appends to
# THEROCK_DEFAULT_CMAKE_VARS, prepares per-subproject injection data, and
# generates flag_settings.json and the provider state file.
# Must be called after all flags are declared and before subprojects are activated.
function(therock_finalize_flags)
  get_property(_all_flags GLOBAL PROPERTY THEROCK_ALL_FLAGS)

  # Phase 1: Create cache variables from stored defaults.
  # This is the single place where THEROCK_FLAG_* cache vars are created,
  # ensuring no set-ordering issues between declare and override.
  foreach(_flag_name ${_all_flags})
    get_property(_default GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_DEFAULT_VALUE)
    get_property(_description GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_DESCRIPTION)
    get_property(_type GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_TYPE)
    if(_type STREQUAL "BOOL")
      set(_cache_type BOOL)
    else()
      set(_cache_type STRING)
    endif()
    set(THEROCK_FLAG_${_flag_name} "${_default}" CACHE ${_cache_type} "${_description}")
    _therock_normalize_flag_value(
      _normalized "${_flag_name}" "${THEROCK_FLAG_${_flag_name}}"
      "THEROCK_FLAG_${_flag_name}"
    )
    if(_type STREQUAL "BOOL")
      if(_normalized STREQUAL "1")
        set(_cache_value ON)
      else()
        set(_cache_value OFF)
      endif()
    else()
      set(_cache_value "${_normalized}")
    endif()
    set(THEROCK_FLAG_${_flag_name} "${_cache_value}"
      CACHE ${_cache_type} "${_description}" FORCE)
    # Propagate the (possibly user-overridden) cache value to the caller's scope.
    set(THEROCK_FLAG_${_flag_name} "${THEROCK_FLAG_${_flag_name}}" PARENT_SCOPE)
  endforeach()

  # Phase 2: Process enabled flags and build JSON and provider state.
  set(_json_entries)
  set(_state_names ${_all_flags})
  set(_state_content
    "set(ROCM_BUILD_FLAGS_PROTOCOL_VERSION ${_THEROCK_BUILD_FLAGS_PROTOCOL_VERSION})\n")
  string(APPEND _state_content "set(ROCM_BUILD_FLAGS_PROVIDER \"TheRock\")\n")
  string(APPEND _state_content "set(ROCM_BUILD_FLAGS_NAMES\n")
  foreach(_state_name ${_state_names})
    string(APPEND _state_content "  ${_state_name}\n")
  endforeach()
  string(APPEND _state_content ")\n")

  foreach(_flag_name ${_all_flags})
    get_property(_type GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_TYPE)
    _therock_normalize_flag_value(
      _normalized "${_flag_name}" "${THEROCK_FLAG_${_flag_name}}"
      "THEROCK_FLAG_${_flag_name}"
    )

    if(_type STREQUAL "BOOL" AND _normalized STREQUAL "1")
      list(APPEND _json_entries "\"${_flag_name}\": true")
    elseif(_type STREQUAL "BOOL")
      list(APPEND _json_entries "\"${_flag_name}\": false")
    else()
      list(APPEND _json_entries "\"${_flag_name}\": ${_normalized}")
    endif()
    string(APPEND _state_content
      "set(ROCM_BUILD_FLAG_${_flag_name}_TYPE \"${_type}\")\n"
      "set(ROCM_BUILD_FLAG_${_flag_name}_VALUE \"${_normalized}\")\n"
    )

    get_property(_global_propagate_flag GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_GLOBAL_PROPAGATE_FLAG)
    if(_global_propagate_flag)
      set_property(GLOBAL APPEND PROPERTY THEROCK_DEFAULT_CMAKE_VARS THEROCK_FLAG_${_flag_name})
    endif()

    if(NOT THEROCK_FLAG_${_flag_name})
      continue()  # Flag is OFF, skip enabled-only propagation processing.
    endif()

    # Process GLOBAL_CMAKE_VARS: set in super-project and add to default vars list.
    get_property(_global_cmake_vars GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_GLOBAL_CMAKE_VARS)
    foreach(_var_pair ${_global_cmake_vars})
      string(FIND "${_var_pair}" "=" _eq_pos)
      if(_eq_pos EQUAL -1)
        message(FATAL_ERROR
          "Flag '${_flag_name}' GLOBAL_CMAKE_VARS entry '${_var_pair}' "
          "must be in VAR=VALUE format"
        )
      endif()
      string(SUBSTRING "${_var_pair}" 0 ${_eq_pos} _var_name)
      math(EXPR _val_start "${_eq_pos} + 1")
      string(SUBSTRING "${_var_pair}" ${_val_start} -1 _var_value)

      # Set in super-project scope.
      set(${_var_name} "${_var_value}" PARENT_SCOPE)
      # Add to the default vars list so it propagates to all subprojects.
      set_property(GLOBAL APPEND PROPERTY THEROCK_DEFAULT_CMAKE_VARS ${_var_name})
    endforeach()

    # Process GLOBAL_CPP_DEFINES.
    get_property(_global_cpp_defines GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_GLOBAL_CPP_DEFINES)
    foreach(_define ${_global_cpp_defines})
      set_property(GLOBAL APPEND PROPERTY THEROCK_FLAG_GLOBAL_CPP_DEFINES "${_define}")
    endforeach()

    # Process per-subproject CMAKE_VARS and CPP_DEFINES.
    get_property(_cmake_vars GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_CMAKE_VARS)
    get_property(_cpp_defines GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_CPP_DEFINES)
    get_property(_sub_projects GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_SUB_PROJECTS)

    foreach(_subproject ${_sub_projects})
      foreach(_var_pair ${_cmake_vars})
        set_property(GLOBAL APPEND PROPERTY
          _THEROCK_SUBPROJECT_FLAG_CMAKE_VARS_${_subproject} "${_var_pair}")
      endforeach()
      foreach(_define ${_cpp_defines})
        set_property(GLOBAL APPEND PROPERTY
          _THEROCK_SUBPROJECT_FLAG_CPP_DEFINES_${_subproject} "${_define}")
      endforeach()
    endforeach()
  endforeach()

  string(APPEND _state_content "set(ROCM_BUILD_FLAGS_STATE_COMPLETE 1)\n")

  # Generate typed settings and provider state in the build directory. file(CONFIGURE)
  # only updates the output when its contents change.
  list(JOIN _json_entries ",\n  " _json_body)
  set(_json_content "{\n  ${_json_body}\n}\n")
  set(_flag_settings_file "${THEROCK_BINARY_DIR}/flag_settings.json")
  file(CONFIGURE OUTPUT "${_flag_settings_file}" CONTENT "${_json_content}" @ONLY)
  set(_build_flags_state_file "${THEROCK_BINARY_DIR}/rocm_build_flags_state.cmake")
  file(CONFIGURE OUTPUT "${_build_flags_state_file}" CONTENT "${_state_content}" @ONLY)
  set(THEROCK_FLAG_SETTINGS_FILE "${_flag_settings_file}" PARENT_SCOPE)
  set(ROCM_BUILD_FLAGS_STATE_FILE "${_build_flags_state_file}" PARENT_SCOPE)
endfunction()

# therock_report_flags
# Reports the status of all declared flags at the end of configure.
function(therock_report_flags)
  get_property(_all_flags GLOBAL PROPERTY THEROCK_ALL_FLAGS)
  if(NOT _all_flags)
    return()
  endif()

  message(STATUS "Build flags:")
  foreach(_flag_name ${_all_flags})
    get_property(_type GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_TYPE)
    if(_type STREQUAL "BOOL" AND THEROCK_FLAG_${_flag_name})
      set(_display_value ON)
    elseif(_type STREQUAL "BOOL")
      set(_display_value OFF)
    else()
      set(_display_value "${THEROCK_FLAG_${_flag_name}}")
    endif()
    message(STATUS
      "  * ${_flag_name} = ${_display_value} "
      "(-DTHEROCK_FLAG_${_flag_name}=${_display_value})"
    )
  endforeach()
endfunction()

# _therock_get_flag_init_contents
# Internal function called from therock_cmake_subproject_activate() to get
# flag-injected content for a specific subproject's project_init.cmake.
# Sets ${out_var} in PARENT_SCOPE with the content to append.
function(_therock_get_flag_init_contents out_var target_name)
  set(_contents "")

  # Global CPP defines (apply to ALL subprojects).
  get_property(_global_cpp_defines GLOBAL PROPERTY THEROCK_FLAG_GLOBAL_CPP_DEFINES)
  if(_global_cpp_defines)
    string(APPEND _contents "\n# Flag system: global CPP defines\n")
    foreach(_define ${_global_cpp_defines})
      string(APPEND _contents "add_compile_definitions(${_define})\n")
    endforeach()
  endif()

  # Per-subproject CMAKE_VARS.
  get_property(_has_cmake_vars GLOBAL PROPERTY _THEROCK_SUBPROJECT_FLAG_CMAKE_VARS_${target_name} SET)
  if(_has_cmake_vars)
    get_property(_cmake_vars GLOBAL PROPERTY _THEROCK_SUBPROJECT_FLAG_CMAKE_VARS_${target_name})
    if(_cmake_vars)
      string(APPEND _contents "\n# Flag system: per-subproject CMAKE vars\n")
      foreach(_var_pair ${_cmake_vars})
        string(FIND "${_var_pair}" "=" _eq_pos)
        string(SUBSTRING "${_var_pair}" 0 ${_eq_pos} _var_name)
        math(EXPR _val_start "${_eq_pos} + 1")
        string(SUBSTRING "${_var_pair}" ${_val_start} -1 _var_value)
        string(APPEND _contents "set(${_var_name} \"${_var_value}\" CACHE STRING \"\" FORCE)\n")
      endforeach()
    endif()
  endif()

  # Per-subproject CPP defines.
  get_property(_has_cpp_defines GLOBAL PROPERTY _THEROCK_SUBPROJECT_FLAG_CPP_DEFINES_${target_name} SET)
  if(_has_cpp_defines)
    get_property(_cpp_defines GLOBAL PROPERTY _THEROCK_SUBPROJECT_FLAG_CPP_DEFINES_${target_name})
    if(_cpp_defines)
      string(APPEND _contents "\n# Flag system: per-subproject CPP defines\n")
      foreach(_define ${_cpp_defines})
        string(APPEND _contents "add_compile_definitions(${_define})\n")
      endforeach()
    endif()
  endif()

  set(${out_var} "${_contents}" PARENT_SCOPE)
endfunction()
