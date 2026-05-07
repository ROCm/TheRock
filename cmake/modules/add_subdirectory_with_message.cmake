# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Adds a subdirectory to the build with a status message,
# and optionally checks for an expected target.
#
# Usage:
#   add_subdirectory_with_message(COMPONENT <component> PREFIX_PATH <prefix> [EXPECT_TARGET <target>])
#
# Arguments:
#   COMPONENT <component>  - Component name (e.g., "mxdatagenerator").
#   PREFIX_PATH <prefix>   - Path prefix (e.g., "shared", "projects").
#   EXPECT_TARGET <target> - Optional target name produced by the subdirectory.
function(add_subdirectory_with_message)
    cmake_parse_arguments(ARG "" "COMPONENT;PREFIX_PATH;EXPECT_TARGET" "" ${ARGN})

    if(NOT ARG_COMPONENT)
        message(FATAL_ERROR "add_subdirectory_with_message: COMPONENT is required")
    endif()
    if(NOT ARG_PREFIX_PATH)
        message(FATAL_ERROR "add_subdirectory_with_message: PREFIX_PATH is required")
    endif()

    set(_subdir_path "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PREFIX_PATH}/${ARG_COMPONENT}")
    file(TO_CMAKE_PATH "${_subdir_path}" _subdir_path)

    list(APPEND CMAKE_MESSAGE_CONTEXT "${ARG_COMPONENT}")

    add_subdirectory("${_subdir_path}")

    if(ARG_EXPECT_TARGET AND NOT TARGET ${ARG_EXPECT_TARGET})
        message(FATAL_ERROR "Expected target ${ARG_EXPECT_TARGET} not found in ${_subdir_path}")
    endif()

    list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endfunction()
