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
#   THEROCK_FLAG_${NAME}: boolean cache variable controlling the flag state
#   Optionally propagates CMAKE variables and CPP defines to all or specific
#   subprojects via the project_init.cmake mechanism.
#
# See docs/development/flags.md for full documentation.

# Global property to track all declared flags.
set_property(GLOBAL PROPERTY THEROCK_ALL_FLAGS)

# therock_declare_flag
# Declares a build flag with optional variable and define propagation.
#
# Arguments:
#   NAME           - Unique flag identifier (creates THEROCK_FLAG_${NAME} cache var)
#   DEFAULT_VALUE  - ON or OFF
#   DESCRIPTION    - Short description for the cache variable
#   ISSUE          - (Optional) Tracking issue URL
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
    ""
    "NAME;DEFAULT_VALUE;DESCRIPTION;ISSUE"
    "GLOBAL_CMAKE_VARS;GLOBAL_CPP_DEFINES;CMAKE_VARS;CPP_DEFINES;SUB_PROJECTS"
  )

  # Validate required arguments.
  if(NOT ARG_NAME)
    message(FATAL_ERROR "therock_declare_flag: NAME is required")
  endif()
  if(NOT DEFINED ARG_DEFAULT_VALUE)
    message(FATAL_ERROR "therock_declare_flag: DEFAULT_VALUE is required for flag ${ARG_NAME}")
  endif()
  if(NOT ARG_DESCRIPTION)
    message(FATAL_ERROR "therock_declare_flag: DESCRIPTION is required for flag ${ARG_NAME}")
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
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_DEFAULT_VALUE "${ARG_DEFAULT_VALUE}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_DESCRIPTION "${ARG_DESCRIPTION}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_GLOBAL_CMAKE_VARS "${ARG_GLOBAL_CMAKE_VARS}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_GLOBAL_CPP_DEFINES "${ARG_GLOBAL_CPP_DEFINES}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_CMAKE_VARS "${ARG_CMAKE_VARS}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_CPP_DEFINES "${ARG_CPP_DEFINES}")
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_SUB_PROJECTS "${ARG_SUB_PROJECTS}")
  if(ARG_ISSUE)
    set_property(GLOBAL PROPERTY _THEROCK_FLAG_${ARG_NAME}_ISSUE "${ARG_ISSUE}")
  endif()
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
  set_property(GLOBAL PROPERTY _THEROCK_FLAG_${flag_name}_DEFAULT_VALUE "${new_default}")
endfunction()

# therock_finalize_flags
# Processes all declared flags: sets global variables, appends to
# THEROCK_DEFAULT_CMAKE_VARS, prepares per-subproject injection data, and
# generates the flag_settings.json file.
# Must be called after all flags are declared and before subprojects are activated.
function(therock_finalize_flags)
  get_property(_all_flags GLOBAL PROPERTY THEROCK_ALL_FLAGS)

  # Phase 1: Create cache variables from stored defaults.
  # This is the single place where THEROCK_FLAG_* cache vars are created,
  # ensuring no set-ordering issues between declare and override.
  foreach(_flag_name ${_all_flags})
    get_property(_default GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_DEFAULT_VALUE)
    get_property(_description GLOBAL PROPERTY _THEROCK_FLAG_${_flag_name}_DESCRIPTION)
    set(THEROCK_FLAG_${_flag_name} "${_default}" CACHE BOOL "${_description}")
    # Propagate the (possibly user-overridden) cache value to the caller's scope.
    set(THEROCK_FLAG_${_flag_name} "${THEROCK_FLAG_${_flag_name}}" PARENT_SCOPE)
  endforeach()

  # Phase 2: Process enabled flags and build JSON.
  set(_json_entries)

  foreach(_flag_name ${_all_flags})
    # Record flag state for JSON output.
    if(THEROCK_FLAG_${_flag_name})
      list(APPEND _json_entries "\"${_flag_name}\": true")
    else()
      list(APPEND _json_entries "\"${_flag_name}\": false")
    endif()

    if(NOT THEROCK_FLAG_${_flag_name})
      continue()  # Flag is OFF, skip propagation processing.
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

  # Generate flag_settings.json in the build directory.
  list(JOIN _json_entries ",\n  " _json_body)
  set(_json_content "{\n  ${_json_body}\n}\n")
  set(_flag_settings_file "${THEROCK_BINARY_DIR}/flag_settings.json")
  file(WRITE "${_flag_settings_file}" "${_json_content}")
  set(THEROCK_FLAG_SETTINGS_FILE "${_flag_settings_file}" PARENT_SCOPE)
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
    if(THEROCK_FLAG_${_flag_name})
      message(STATUS "  * ${_flag_name} = ON (-DTHEROCK_FLAG_${_flag_name}=ON)")
    else()
      message(STATUS "  * ${_flag_name} = OFF (-DTHEROCK_FLAG_${_flag_name}=OFF)")
    endif()
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
