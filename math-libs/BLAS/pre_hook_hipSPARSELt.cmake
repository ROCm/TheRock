# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Tensile just uses the system path to find most of its tools and it does this
# in the build phase. Rather than tunneling everything through manually, we
# just explicitly set up the path to include our toolchain ROCM and LLVM
# tools. This kind of reacharound is not great but the project is old, so
# c'est la vie.

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

# Tensile source kernels require explicit xnack qualifiers for CDNA targets.
# See pre_hook_rocBLAS.cmake for full explanation.
# hipSPARSELt uses TensileLite with the same target restrictions as hipBLASLt:
# gfx942:xnack+ and gfx950:xnack+ only; gfx90a accepts both xnack variants.
if(NOT THEROCK_SANITIZER)
  set(_tensile_xnack_targets)
  foreach(_t IN LISTS GPU_TARGETS)
    if("${_t}" STREQUAL "gfx90a")
      list(APPEND _tensile_xnack_targets "${_t}:xnack-" "${_t}:xnack+")
    elseif("${_t}" STREQUAL "gfx942" OR "${_t}" STREQUAL "gfx950")
      list(APPEND _tensile_xnack_targets "${_t}:xnack+")
    else()
      list(APPEND _tensile_xnack_targets "${_t}")
    endif()
  endforeach()
  if(NOT "${_tensile_xnack_targets}" STREQUAL "${GPU_TARGETS}")
    message(STATUS "hipSPARSELt: expanding Tensile GPU targets with xnack variants: ${_tensile_xnack_targets}")
    set(GPU_TARGETS "${_tensile_xnack_targets}" CACHE STRING "GPU targets" FORCE)
    set(AMDGPU_TARGETS "${_tensile_xnack_targets}" CACHE STRING "AMDGPU targets" FORCE)
  endif()
endif()
