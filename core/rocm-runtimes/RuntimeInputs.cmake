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
    elseif(_entry MATCHES "(CMakeLists\\.txt|Cargo\\.lock|/\\.cargo/config|/tests/fixtures/.*|\\.(rs|c|cc|cpp|h|hh|hpp|hxx|inc|S|s|map|bin|cl|hip|cmake|py|in|toml))$")
      list(APPEND _inputs "${_entry}")
    endif()
  endforeach()
  set("${out_var}" "${_inputs}" PARENT_SCOPE)
endfunction()

# Record source membership and native Rust inputs before declaring the producer.
# Sources also become outer build prerequisites; a fingerprint alone does not
# invalidate retained Ninja build and test stamps.
function(therock_prepare_runtime_inputs args_out state_out depends_out sources_out)
  set(ROCM_RUNTIMES_CARGO "" CACHE FILEPATH "Cargo executable (empty uses upstream discovery)")
  set(ROCM_RUNTIMES_RUSTC "" CACHE FILEPATH "Rust compiler (empty uses upstream discovery)")
  set(ROCM_RUNTIMES_LLD "" CACHE FILEPATH "Rust linker (empty uses TheRock's compiler toolchain)")
  set(ROCM_RUNTIMES_CARGO_HOME "" CACHE PATH "Prepared Cargo home (empty uses upstream's isolated home)")
  set(ROCM_RUNTIMES_CARGO_CONFIG "" CACHE FILEPATH "Prepared Cargo configuration")
  set(ROCM_RUNTIMES_CARGO_JOBS "" CACHE STRING "Cargo parallel-job limit")
  set(_args)
  set(_depends)
  set(_state "")
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

  foreach(_name CARGO RUSTC LLD CARGO_HOME CARGO_CONFIG CARGO_JOBS)
    set(_value "${ROCM_RUNTIMES_${_name}}")
    if(NOT _name STREQUAL "CARGO_JOBS" AND NOT _value STREQUAL "")
      # Untyped -D paths are normalized by their cache declarations above.
      # Any remaining relative paths use the top-level source directory.
      cmake_path(ABSOLUTE_PATH _value BASE_DIRECTORY "${CMAKE_SOURCE_DIR}" NORMALIZE)
      set_property(CACHE ROCM_RUNTIMES_${_name} PROPERTY VALUE "${_value}")
    endif()
    string(APPEND _state "${_name}=${_value}\n")
    if(_value STREQUAL "")
      # Remove retained child overrides so upstream can initialize its defaults.
      list(APPEND _args "-UROCM_RUNTIMES_${_name}")
    else()
      string(REPLACE ";" "\\;" _argument "${_value}")
      list(APPEND _args "-DROCM_RUNTIMES_${_name}=${_argument}")
    endif()
    if(_name MATCHES "^(CARGO|RUSTC|LLD|CARGO_CONFIG)$" AND NOT _value STREQUAL "")
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
