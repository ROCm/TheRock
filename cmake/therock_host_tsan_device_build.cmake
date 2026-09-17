# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

function(_therock_without_debug_info out_var flags)
  string(REGEX REPLACE "(^|[ ])-g([0-9]+)?([ ]|$)" " " flags "${flags}")
  string(REGEX REPLACE "(^|[ ])-gdwarf-[0-9]+([ ]|$)" " " flags "${flags}")
  string(REGEX REPLACE "[ ]+" " " flags "${flags}")
  string(STRIP "${flags}" flags)
  set(${out_var} "${flags}" PARENT_SCOPE)
endfunction()

# Removes only debug-info generation from a HOST_TSAN subproject while retaining
# compiler resource paths, host TSAN instrumentation, and all device targets.
# Handles both fresh configures and retried inner CMake caches.
function(therock_host_tsan_strip_debug_info)
  if(NOT THEROCK_SANITIZER STREQUAL "HOST_TSAN")
    return()
  endif()

  foreach(_flags_var CMAKE_C_FLAGS_INIT CMAKE_CXX_FLAGS_INIT)
    _therock_without_debug_info(_flags "${${_flags_var}}")
    set(${_flags_var} "${_flags}" PARENT_SCOPE)
  endforeach()

  foreach(_flags_var CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    if(DEFINED CACHE{${_flags_var}})
      _therock_without_debug_info(_flags "${${_flags_var}}")
      set(${_flags_var} "${_flags}" CACHE STRING
        "Flags used by the C/C++ compiler" FORCE)
    endif()
  endforeach()
endfunction()
