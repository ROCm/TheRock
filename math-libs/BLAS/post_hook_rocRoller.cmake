# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Post-hook for rocRoller to wrap build-time executables with ASAN runtime preload.
#
# rocRoller's GPUArchitectureGenerator runs during build via add_custom_command().
# When rocRoller is built without ASAN but links to ASAN-instrumented dependencies
# (hip-clr, etc.), the executable needs LD_PRELOAD to avoid "ASan runtime does not
# come first" errors.
#
# CMAKE_CROSSCOMPILING_EMULATOR doesn't work with add_custom_command(), so we wrap
# the executable itself with a shell script after it's built.

if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  if(NOT DEFINED THEROCK_SANITIZER_LAUNCHER)
    message(FATAL_ERROR "rocRoller post-hook: THEROCK_SANITIZER_LAUNCHER not defined")
  endif()

  # Find GPUArchitectureGenerator in the list of executable targets
  if("GPUArchitectureGenerator" IN_LIST THEROCK_EXECUTABLE_TARGETS)
    message(STATUS "rocRoller: Wrapping GPUArchitectureGenerator with ASAN runtime preload")
    
    # Convert THEROCK_SANITIZER_LAUNCHER list to space-separated string for shell script
    string(JOIN " " _launcher_str ${THEROCK_SANITIZER_LAUNCHER})
    
    # Create a helper script that will generate the wrapper at build time
    set(_wrapper_gen_script "${CMAKE_CURRENT_BINARY_DIR}/generate_wrapper.cmake")
    file(WRITE "${_wrapper_gen_script}" "
# Auto-generated script to wrap GPUArchitectureGenerator with ASAN runtime
file(RENAME \"\${REAL_EXE}\" \"\${REAL_EXE}.real\")

file(WRITE \"\${REAL_EXE}\" \"#!/bin/bash
# Auto-generated wrapper for ASAN runtime preload
exec ${_launcher_str} \\\"\${REAL_EXE}.real\\\" \\\"\\\$@\\\"
\")

file(CHMOD \"\${REAL_EXE}\" PERMISSIONS 
  OWNER_READ OWNER_WRITE OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE)
")
    
    # Add POST_BUILD command to wrap the executable
    add_custom_command(TARGET GPUArchitectureGenerator POST_BUILD
      COMMENT "Wrapping GPUArchitectureGenerator with ASAN runtime launcher"
      COMMAND ${CMAKE_COMMAND}
        -DREAL_EXE="$<TARGET_FILE:GPUArchitectureGenerator>"
        -P "${_wrapper_gen_script}"
      VERBATIM
    )
  else()
    message(WARNING "rocRoller post-hook: GPUArchitectureGenerator target not found in THEROCK_EXECUTABLE_TARGETS")
  endif()
endif()
