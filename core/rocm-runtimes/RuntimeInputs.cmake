# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Cargo output trees are not source inputs. Walking one directory at a time
# keeps their contents out of CONFIGURE_DEPENDS while detecting new source dirs.
function(_therock_collect_runtime_inputs out_var source_dir)
  file(GLOB _entries LIST_DIRECTORIES TRUE CONFIGURE_DEPENDS "${source_dir}/*" "${source_dir}/.cargo")
  set(_inputs)
  foreach(_entry IN LISTS _entries)
    if(IS_DIRECTORY "${_entry}")
      cmake_path(GET _entry FILENAME _name)
      if(NOT _name MATCHES "^(target|cargo-home|__pycache__|\\.git)$")
        _therock_collect_runtime_inputs(_children "${_entry}")
        list(APPEND _inputs ${_children})
      endif()
    else()
      list(APPEND _inputs "${_entry}")
    endif()
  endforeach()
  set("${out_var}" "${_inputs}" PARENT_SCOPE)
endfunction()

# Record source membership and native Rust inputs before declaring the producer.
# Sources also become outer build prerequisites; a fingerprint alone does not
# invalidate retained Ninja build and test stamps.
function(therock_prepare_runtime_inputs args_out state_out depends_out sources_out)
  set(_args)
  set(_depends)
  set(_state "runtime-inputs-v2\n")
  _therock_collect_runtime_inputs(_sources "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/runtimes")
  list(APPEND _sources "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
  list(SORT _sources)
  string(APPEND _state "SOURCES=${_sources}\n")
  set(_configure_sources ${_sources})
  list(FILTER _configure_sources INCLUDE REGEX "(CMakeLists\\.txt|\\.(cmake|py|h|hh|hpp|hxx|inc|in|toml)|Cargo\\.lock|/\\.cargo/config)$")

  # Upstream captures these values into its generated Cargo command at configure.
  foreach(_name RUSTFLAGS CARGO_ENCODED_RUSTFLAGS PATH)
    if(DEFINED ENV{${_name}})
      string(APPEND _state "ENV_${_name}_DEFINED=ON\nENV_${_name}=$ENV{${_name}}\n")
    else()
      string(APPEND _state "ENV_${_name}_DEFINED=OFF\n")
    endif()
  endforeach()

  # CMake's invocation directory can differ from its source/build directories.
  # Persist normalized inputs so Ninja regeneration from another cwd is stable.
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import os; print(os.getcwd())"
    OUTPUT_VARIABLE _invocation_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  foreach(_tool CARGO RUSTC)
    string(TOLOWER "${_tool}" _program)
    if(NOT ROCM_RUNTIMES_${_tool})
      set(ROCM_RUNTIMES_${_tool} "ROCM_RUNTIMES_${_tool}-NOTFOUND")
      find_program(ROCM_RUNTIMES_${_tool} NAMES "${_program}" REQUIRED)
    endif()
  endforeach()
  foreach(_name CARGO RUSTC LLD CARGO_HOME CARGO_CONFIG CARGO_JOBS)
    set(_reset FALSE)
    if(DEFINED ROCM_RUNTIMES_${_name})
      set(_value "${ROCM_RUNTIMES_${_name}}")
    else()
      set(_value "")
    endif()
    if(_name MATCHES "^(CARGO_HOME|LLD)$" AND _value STREQUAL "")
      # An explicitly empty home selects upstream's isolated cargo-home.
      # An absent home or linker permits child default initialization.
      set(_reset TRUE)
    endif()
    if(_name MATCHES "^(CARGO_HOME|CARGO_CONFIG)$" AND NOT _value STREQUAL "")
      cmake_path(ABSOLUTE_PATH _value BASE_DIRECTORY "${_invocation_dir}" NORMALIZE)
      if(DEFINED ROCM_RUNTIMES_${_name})
        set(ROCM_RUNTIMES_${_name} "${_value}" CACHE STRING "Runtime ${_name}" FORCE)
        set(ROCM_RUNTIMES_${_name} "${_value}" PARENT_SCOPE)
      endif()
    endif()
    string(APPEND _state "${_name}=${_value}\n")
    if(_reset)
      list(APPEND _args "-UROCM_RUNTIMES_${_name}")
    else()
      string(REPLACE ";" "\\;" _argument "${_value}")
      list(APPEND _args "-DROCM_RUNTIMES_${_name}=${_argument}")
    endif()
    if(_name MATCHES "^(CARGO|RUSTC|LLD|CARGO_CONFIG)$" AND NOT _value STREQUAL "")
      if(NOT IS_ABSOLUTE "${_value}" OR NOT EXISTS "${_value}" OR IS_DIRECTORY "${_value}")
        message(FATAL_ERROR "ROCM_RUNTIMES_${_name} must be an existing absolute file path")
      endif()
      file(SHA256 "${_value}" _sha256)
      string(APPEND _state "${_name}_SHA256=${_sha256}\n")
      list(APPEND _depends "${_value}")
    endif()
  endforeach()
  set(_state_path "${CMAKE_CURRENT_BINARY_DIR}/rocm-runtimes-inputs.txt")
  file(CONFIGURE OUTPUT "${_state_path}" CONTENT "${_state}" @ONLY)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_depends})
  set("${args_out}" "${_args}" PARENT_SCOPE)
  set("${state_out}" "${_state_path}" PARENT_SCOPE)
  set("${depends_out}" "${_state_path};${_depends};${_configure_sources}" PARENT_SCOPE)
  set("${sources_out}" "${_sources}" PARENT_SCOPE)
endfunction()
