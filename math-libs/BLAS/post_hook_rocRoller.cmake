# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Post-hook for rocRoller to wrap build-time executables with ASAN runtime preload.
#
# rocRoller's GPUArchitectureGenerator runs during build via add_custom_command().
# When rocRoller is built without ASAN but links to ASAN-instrumented dependencies
# (hip-clr, etc.), the executable needs LD_PRELOAD to avoid "ASan runtime does not
# come first" errors.
#
# We rename the target's output and create a wrapper script at configure time.

if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  if(NOT DEFINED THEROCK_SANITIZER_LAUNCHER)
    message(FATAL_ERROR "rocRoller post-hook: THEROCK_SANITIZER_LAUNCHER not defined")
  endif()

  # Find GPUArchitectureGenerator in the list of executable targets
  if("GPUArchitectureGenerator" IN_LIST THEROCK_EXECUTABLE_TARGETS)
    message(STATUS "rocRoller: Setting up ASAN wrapper for GPUArchitectureGenerator")
    
    # Convert THEROCK_SANITIZER_LAUNCHER list to space-separated string
    string(JOIN " " _launcher_str ${THEROCK_SANITIZER_LAUNCHER})
    
    # Modify the target to output with .real suffix
    set_target_properties(GPUArchitectureGenerator PROPERTIES
      OUTPUT_NAME "GPUArchitectureGenerator.real"
    )
    
    # Determine where the executable will be built
    get_target_property(_output_dir GPUArchitectureGenerator RUNTIME_OUTPUT_DIRECTORY)
    if(NOT _output_dir)
      # Use the default location from the binary dir
      set(_output_dir "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitectureGenerator")
    endif()
    
    set(_wrapper_path "${_output_dir}/GPUArchitectureGenerator")
    set(_real_exe_path "${_output_dir}/GPUArchitectureGenerator.real")
    
    # Create the wrapper script at configure time
    file(WRITE "${_wrapper_path}" "#!/bin/bash\n")
    file(APPEND "${_wrapper_path}" "# Auto-generated ASAN wrapper for GPUArchitectureGenerator\n")
    file(APPEND "${_wrapper_path}" "DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n")
    file(APPEND "${_wrapper_path}" "exec ${_launcher_str} \"$DIR/GPUArchitectureGenerator.real\" \"$@\"\n")
    
    file(CHMOD "${_wrapper_path}"
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                  GROUP_READ GROUP_EXECUTE
                  WORLD_READ WORLD_EXECUTE)
    
    message(STATUS "rocRoller: Created ASAN wrapper at ${_wrapper_path}")
  else()
    message(WARNING "rocRoller post-hook: GPUArchitectureGenerator target not found in THEROCK_EXECUTABLE_TARGETS")
  endif()
endif()
