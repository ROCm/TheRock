# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Tensile just uses the system path to find most of its tools and it does this
# in the build phase. Rather than tunneling everything through manually, we
# just explicitly set up the path to include our toolchain ROCM and LLVM
# tools. This kind of reacharound is not great but the project is old, so
# c'est la vie.

# Tensile launches a separate worker pool, so Ninja's job limit does not bound
# it. Forward the measured CSAN CI budget through TensileLite's supported cache
# option, which emits --jobs for TensileCreateLibrary. Keep an explicit smaller
# positive Tensile limit when one was requested.
if(THEROCK_SANITIZER STREQUAL "CSAN" AND DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
  if(NOT "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "CSAN requires a positive CMAKE_BUILD_PARALLEL_LEVEL when set")
  endif()
  if(NOT TENSILELITE_BUILD_PARALLEL_LEVEL MATCHES "^[1-9][0-9]*$" OR
      TENSILELITE_BUILD_PARALLEL_LEVEL GREATER "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
    set(TENSILELITE_BUILD_PARALLEL_LEVEL "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" CACHE STRING
      "CPU workers for Tensile device-library generation" FORCE)
  endif()
endif()

block(SCOPE_FOR VARIABLES)
  if(NOT THEROCK_TOOLCHAIN_ROOT)
    message(FATAL_ERROR "As a sub-project, THEROCK_TOOLCHAIN_ROOT should have been defined and was not")
  endif()
  if(WIN32)
    set(ps ";")
  else()
    set(ps ":")
  endif()
  set(new_path)
  foreach(path_item ${CMAKE_PROGRAM_PATH})
    string(APPEND new_path "${path_item}${ps}")
  endforeach()
  string(APPEND new_path "$ENV{PATH}")

  set(ENV{PATH} "${new_path}")
  message(STATUS "Augmented toolchain PATH=$ENV{PATH}")
endblock()
