# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Canonical source: TheRock/cmake/ROCMBuildFlags.cmake
#
# This module is copied verbatim into the ROCm super-repositories. Keep it
# compatible with CMake 3.7 and free of external module dependencies.

if(DEFINED ROCM_BUILD_FLAGS_MODULE_VERSION)
    return()
endif()
set(ROCM_BUILD_FLAGS_MODULE_VERSION 1)

# Raises a consistently prefixed fatal error.
function(_rocm_build_flags_fail)
    set(_message_text)
    foreach(_message_part ${ARGV})
        string(APPEND _message_text "${_message_part}")
    endforeach()
    message(FATAL_ERROR "ROCMBuildFlags: ${_message_text}")
endfunction()

# Normalizes a supported CMake boolean spelling to 0 or 1.
function(_rocm_build_flags_normalize_bool out_var value context)
    string(TOUPPER "${value}" _value_upper)
    if(
        _value_upper STREQUAL "1"
        OR _value_upper STREQUAL "ON"
        OR _value_upper STREQUAL "YES"
        OR _value_upper STREQUAL "TRUE"
        OR _value_upper STREQUAL "Y"
    )
        set(${out_var} "1" PARENT_SCOPE)
    elseif(
        _value_upper STREQUAL "0"
        OR _value_upper STREQUAL "OFF"
        OR _value_upper STREQUAL "NO"
        OR _value_upper STREQUAL "FALSE"
        OR _value_upper STREQUAL "N"
    )
        set(${out_var} "0" PARENT_SCOPE)
    else()
        _rocm_build_flags_fail(
          "${context} has invalid BOOL value '${value}'; use ON/OFF, TRUE/FALSE, "
          "YES/NO, Y/N, or 1/0"
        )
    endif()
endfunction()

# Validates and returns a canonical signed base-10 integer.
function(_rocm_build_flags_normalize_integer out_var value context)
    if(NOT "${value}" MATCHES "^(0|-?[1-9][0-9]*)$")
        _rocm_build_flags_fail(
          "${context} has invalid INTEGER value '${value}'; use canonical signed "
          "base-10 spelling without a leading plus sign or leading zeroes"
        )
    endif()
    set(${out_var} "${value}" PARENT_SCOPE)
endfunction()

# Validates a protocol flag name.
function(_rocm_build_flags_validate_name value context)
    if(NOT "${value}" MATCHES "^[A-Z][A-Z0-9_]*$")
        _rocm_build_flags_fail(
          "${context} '${value}' must use uppercase letters, digits, and "
          "underscores, and must start with a letter"
        )
    endif()
endfunction()

# Validates all metadata and values loaded from provider state.
function(_rocm_build_flags_validate_state)
    if(NOT ROCM_BUILD_FLAGS_PROTOCOL_VERSION STREQUAL "1")
        _rocm_build_flags_fail(
          "state file '${ROCM_BUILD_FLAGS_STATE_FILE}' uses unsupported protocol "
          "version '${ROCM_BUILD_FLAGS_PROTOCOL_VERSION}'"
        )
    endif()
    if("${ROCM_BUILD_FLAGS_PROVIDER}" STREQUAL "")
        _rocm_build_flags_fail(
          "state file '${ROCM_BUILD_FLAGS_STATE_FILE}' has no provider"
        )
    endif()
    if(NOT ROCM_BUILD_FLAGS_STATE_COMPLETE STREQUAL "1")
        _rocm_build_flags_fail(
          "state file '${ROCM_BUILD_FLAGS_STATE_FILE}' is incomplete"
        )
    endif()
    if("${ROCM_BUILD_FLAGS_NAMES}" STREQUAL "")
        _rocm_build_flags_fail(
          "state file '${ROCM_BUILD_FLAGS_STATE_FILE}' has no flag names"
        )
    endif()

    set(_seen_names)
    foreach(_name ${ROCM_BUILD_FLAGS_NAMES})
        _rocm_build_flags_validate_name("${_name}" "state flag name")
        if("${_name}" IN_LIST _seen_names)
            _rocm_build_flags_fail(
              "state file '${ROCM_BUILD_FLAGS_STATE_FILE}' repeats flag '${_name}'"
            )
        endif()
        list(APPEND _seen_names "${_name}")

        if(NOT DEFINED ROCM_BUILD_FLAG_${_name}_TYPE)
            _rocm_build_flags_fail("state flag '${_name}' has no TYPE")
        endif()
        if(NOT DEFINED ROCM_BUILD_FLAG_${_name}_VALUE)
            _rocm_build_flags_fail("state flag '${_name}' has no VALUE")
        endif()
        set(_type "${ROCM_BUILD_FLAG_${_name}_TYPE}")
        set(_value "${ROCM_BUILD_FLAG_${_name}_VALUE}")
        if(_type STREQUAL "BOOL")
            if(NOT _value STREQUAL "0" AND NOT _value STREQUAL "1")
                _rocm_build_flags_fail(
                  "state flag '${_name}' has non-canonical BOOL value '${_value}'"
                )
            endif()
        elseif(_type STREQUAL "INTEGER")
            _rocm_build_flags_normalize_integer(
              _unused "${_value}" "state flag '${_name}'"
            )
        else()
            _rocm_build_flags_fail(
              "state flag '${_name}' has unsupported type '${_type}'"
            )
        endif()
    endforeach()
endfunction()

set(_ROCM_BUILD_FLAGS_INTEGRATED 0)
if(
    DEFINED ROCM_BUILD_FLAGS_STATE_FILE
    AND NOT "${ROCM_BUILD_FLAGS_STATE_FILE}" STREQUAL ""
)
    if(NOT IS_ABSOLUTE "${ROCM_BUILD_FLAGS_STATE_FILE}")
        _rocm_build_flags_fail(
          "ROCM_BUILD_FLAGS_STATE_FILE must be an absolute path: "
          "'${ROCM_BUILD_FLAGS_STATE_FILE}'"
        )
    endif()
    if(NOT EXISTS "${ROCM_BUILD_FLAGS_STATE_FILE}")
        _rocm_build_flags_fail(
          "state file does not exist: '${ROCM_BUILD_FLAGS_STATE_FILE}'"
        )
    endif()

    unset(ROCM_BUILD_FLAGS_PROTOCOL_VERSION)
    unset(ROCM_BUILD_FLAGS_PROVIDER)
    unset(ROCM_BUILD_FLAGS_NAMES)
    unset(ROCM_BUILD_FLAGS_STATE_COMPLETE)
    include("${ROCM_BUILD_FLAGS_STATE_FILE}")
    _rocm_build_flags_validate_state()
    set(_ROCM_BUILD_FLAGS_INTEGRATED 1)
endif()

# Validates the parsed rocm_resolve_build_flag arguments.
function(_rocm_build_flags_validate_request)
    foreach(
        _required_arg
        NAME
        TYPE
        CACHE_VARIABLE
        DEFAULT_VALUE
        DESCRIPTION
        OUTPUT_VARIABLE
    )
        if(
            NOT DEFINED ARG_${_required_arg}
            OR "${ARG_${_required_arg}}" STREQUAL ""
        )
            _rocm_build_flags_fail(
              "rocm_resolve_build_flag requires ${_required_arg}"
            )
        endif()
    endforeach()
    if(ARG_UNPARSED_ARGUMENTS)
        _rocm_build_flags_fail(
          "rocm_resolve_build_flag received unexpected arguments: "
          "${ARG_UNPARSED_ARGUMENTS}"
        )
    endif()

    _rocm_build_flags_validate_name("${ARG_NAME}" "flag NAME")
    if(NOT "${ARG_CACHE_VARIABLE}" MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        _rocm_build_flags_fail(
          "CACHE_VARIABLE '${ARG_CACHE_VARIABLE}' is not a CMake identifier"
        )
    endif()
    if(NOT "${ARG_OUTPUT_VARIABLE}" MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        _rocm_build_flags_fail(
          "OUTPUT_VARIABLE '${ARG_OUTPUT_VARIABLE}' is not a CMake identifier"
        )
    endif()
    if(NOT ARG_TYPE STREQUAL "BOOL" AND NOT ARG_TYPE STREQUAL "INTEGER")
        _rocm_build_flags_fail(
          "flag '${ARG_NAME}' TYPE must be BOOL or INTEGER, not '${ARG_TYPE}'"
        )
    endif()
    if(ARG_TYPE STREQUAL "BOOL" AND ARG_VALID_VALUES)
        _rocm_build_flags_fail(
          "flag '${ARG_NAME}' uses VALID_VALUES, which is INTEGER-only"
        )
    endif()

    foreach(_valid_value ${ARG_VALID_VALUES})
        _rocm_build_flags_normalize_integer(
          _unused "${_valid_value}" "flag '${ARG_NAME}' VALID_VALUES"
        )
    endforeach()
    if(ARG_TYPE STREQUAL "BOOL")
        _rocm_build_flags_normalize_bool(
          _unused "${ARG_DEFAULT_VALUE}" "flag '${ARG_NAME}' DEFAULT_VALUE"
        )
    else()
        _rocm_build_flags_normalize_integer(
          _normalized_default "${ARG_DEFAULT_VALUE}"
          "flag '${ARG_NAME}' DEFAULT_VALUE"
        )
        if(
            NOT "${ARG_VALID_VALUES}" STREQUAL ""
            AND NOT "${_normalized_default}" IN_LIST ARG_VALID_VALUES
        )
            _rocm_build_flags_fail(
              "flag '${ARG_NAME}' DEFAULT_VALUE '${ARG_DEFAULT_VALUE}' is not one of: "
              "${ARG_VALID_VALUES}"
            )
        endif()
    endif()
endfunction()

# Gets an authoritative value from provider state.
function(_rocm_build_flags_get_integrated_value out_var)
    get_property(
        _cache_variable_is_set
        CACHE "${ARG_CACHE_VARIABLE}"
        PROPERTY TYPE
        SET
    )
    if(_cache_variable_is_set)
        _rocm_build_flags_fail(
          "integrated flag '${ARG_NAME}' is authoritative, but project cache "
          "variable '${ARG_CACHE_VARIABLE}' is already set"
        )
    endif()

    if(NOT "${ARG_NAME}" IN_LIST ROCM_BUILD_FLAGS_NAMES)
        _rocm_build_flags_fail(
          "provider '${ROCM_BUILD_FLAGS_PROVIDER}' does not define flag "
          "'${ARG_NAME}'"
        )
    endif()
    set(_provider_type "${ROCM_BUILD_FLAG_${ARG_NAME}_TYPE}")
    if(NOT _provider_type STREQUAL ARG_TYPE)
        _rocm_build_flags_fail(
          "flag '${ARG_NAME}' expects type '${ARG_TYPE}', but provider "
          "'${ROCM_BUILD_FLAGS_PROVIDER}' supplies '${_provider_type}'"
        )
    endif()
    set(${out_var} "${ROCM_BUILD_FLAG_${ARG_NAME}_VALUE}" PARENT_SCOPE)
endfunction()

# Gets a project-specific value from the standalone cache.
function(_rocm_build_flags_get_standalone_value out_var)
    if(ARG_TYPE STREQUAL "BOOL")
        set(_cache_type BOOL)
    else()
        set(_cache_type STRING)
    endif()
    set(${ARG_CACHE_VARIABLE}
        "${ARG_DEFAULT_VALUE}"
        CACHE ${_cache_type}
        "${ARG_DESCRIPTION}"
    )
    get_property(_value CACHE "${ARG_CACHE_VARIABLE}" PROPERTY VALUE)
    set(${out_var} "${_value}" PARENT_SCOPE)
endfunction()

# Validates and normalizes a resolved value.
function(_rocm_build_flags_normalize_resolved_value out_var value)
    if(ARG_TYPE STREQUAL "BOOL")
        _rocm_build_flags_normalize_bool(
          _normalized "${value}" "flag '${ARG_NAME}'"
        )
    else()
        _rocm_build_flags_normalize_integer(
          _normalized "${value}" "flag '${ARG_NAME}'"
        )
        if(
            NOT "${ARG_VALID_VALUES}" STREQUAL ""
            AND NOT "${_normalized}" IN_LIST ARG_VALID_VALUES
        )
            _rocm_build_flags_fail(
              "flag '${ARG_NAME}' value '${_normalized}' is not one of: "
              "${ARG_VALID_VALUES}"
            )
        endif()
    endif()
    set(${out_var} "${_normalized}" PARENT_SCOPE)
endfunction()

# Resolves one typed flag from provider state or a standalone project cache.
function(rocm_resolve_build_flag)
    cmake_parse_arguments(
        PARSE_ARGV
        0
        ARG
        ""
        "NAME;TYPE;CACHE_VARIABLE;DEFAULT_VALUE;DESCRIPTION;OUTPUT_VARIABLE"
        "VALID_VALUES"
    )
    _rocm_build_flags_validate_request()
    if(_ROCM_BUILD_FLAGS_INTEGRATED)
        _rocm_build_flags_get_integrated_value(_value)
    else()
        _rocm_build_flags_get_standalone_value(_value)
    endif()
    _rocm_build_flags_normalize_resolved_value(_value "${_value}")
    set(${ARG_OUTPUT_VARIABLE} "${_value}" PARENT_SCOPE)
endfunction()
