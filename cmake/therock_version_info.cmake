# therock_version_info.cmake
# Centralized version parsing utilities for TheRock build system.
# Provides functions to parse version information from JSON and line-based files.

# Function: therock_parse_version_file
# Parses version information from a file and sets major/minor/patch version variables.
#
# Arguments:
#   FILE <path>     - Path to the version file
#   TYPE <type>     - Type of version file: JSON or LIST
#   PREFIX <prefix> - Prefix for output variables (e.g., ROCM, THEROCK_HIP)
#
# Output Variables (set in PARENT_SCOPE):
#   ${PREFIX}_MAJOR_VERSION - Major version number
#   ${PREFIX}_MINOR_VERSION - Minor version number
#   ${PREFIX}_PATCH_VERSION - Patch version number
#
# Example Usage:
#   therock_parse_version_file(
#     FILE "${THEROCK_SOURCE_DIR}/version.json"
#     TYPE JSON
#     PREFIX ROCM
#   )
#   # Sets: ROCM_MAJOR_VERSION, ROCM_MINOR_VERSION, ROCM_PATCH_VERSION
#
function(therock_parse_version_file)
  cmake_parse_arguments(ARG "" "FILE;TYPE;PREFIX" "" ${ARGN})

  # Validate required arguments
  if(NOT ARG_FILE)
    message(FATAL_ERROR "therock_parse_version_file: FILE argument is required")
  endif()
  if(NOT ARG_TYPE)
    message(FATAL_ERROR "therock_parse_version_file: TYPE argument is required")
  endif()
  if(NOT ARG_PREFIX)
    message(FATAL_ERROR "therock_parse_version_file: PREFIX argument is required")
  endif()

  # Check file exists
  if(NOT EXISTS "${ARG_FILE}")
    message(FATAL_ERROR "Version file not found: ${ARG_FILE}")
  endif()

  # Parse based on type
  if(ARG_TYPE STREQUAL "JSON")
    # Parse JSON version file
    block(PROPAGATE ${ARG_PREFIX}_MAJOR_VERSION ${ARG_PREFIX}_MINOR_VERSION ${ARG_PREFIX}_PATCH_VERSION)
      file(READ "${ARG_FILE}" VERSION_JSON_STRING)
      string(JSON VERSION_STRING GET ${VERSION_JSON_STRING} rocm-version)
      string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" VERSION_MATCH ${VERSION_STRING})

      if(VERSION_MATCH)
        set(${ARG_PREFIX}_MAJOR_VERSION ${CMAKE_MATCH_1})
        set(${ARG_PREFIX}_MINOR_VERSION ${CMAKE_MATCH_2})
        set(${ARG_PREFIX}_PATCH_VERSION ${CMAKE_MATCH_3})
        message(STATUS "${ARG_PREFIX} version: ${${ARG_PREFIX}_MAJOR_VERSION}.${${ARG_PREFIX}_MINOR_VERSION}.${${ARG_PREFIX}_PATCH_VERSION}")
      else()
        message(FATAL_ERROR "Failed to parse ${ARG_PREFIX} version from ${ARG_FILE}")
      endif()
    endblock()

  elseif(ARG_TYPE STREQUAL "LIST")
    # Parse line-based version file (e.g., HIP VERSION file)
    block(SCOPE_FOR VARIABLES PROPAGATE ${ARG_PREFIX}_MAJOR_VERSION ${ARG_PREFIX}_MINOR_VERSION ${ARG_PREFIX}_PATCH_VERSION)
      file(STRINGS "${ARG_FILE}" VERSION_LIST REGEX "^[0-9]+")

      list(LENGTH VERSION_LIST VERSION_LIST_LENGTH)
      if(VERSION_LIST_LENGTH LESS 3)
        message(FATAL_ERROR "Invalid version file format in ${ARG_FILE}: expected at least 3 numeric lines")
      endif()

      list(GET VERSION_LIST 0 ${ARG_PREFIX}_MAJOR_VERSION)
      list(GET VERSION_LIST 1 ${ARG_PREFIX}_MINOR_VERSION)
      list(GET VERSION_LIST 2 ${ARG_PREFIX}_PATCH_VERSION)
      message(STATUS "${ARG_PREFIX} version: ${${ARG_PREFIX}_MAJOR_VERSION}.${${ARG_PREFIX}_MINOR_VERSION}.${${ARG_PREFIX}_PATCH_VERSION}")
    endblock()

  else()
    message(FATAL_ERROR "therock_parse_version_file: Unknown TYPE '${ARG_TYPE}' (must be JSON or LIST)")
  endif()

  # Propagate to parent scope
  set(${ARG_PREFIX}_MAJOR_VERSION ${${ARG_PREFIX}_MAJOR_VERSION} PARENT_SCOPE)
  set(${ARG_PREFIX}_MINOR_VERSION ${${ARG_PREFIX}_MINOR_VERSION} PARENT_SCOPE)
  set(${ARG_PREFIX}_PATCH_VERSION ${${ARG_PREFIX}_PATCH_VERSION} PARENT_SCOPE)
endfunction()
