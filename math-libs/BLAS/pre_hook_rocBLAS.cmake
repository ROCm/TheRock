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
    set(PS ";")
  else()
    set(PS ":")
  endif()
  set(CURRENT_PATH "$ENV{PATH}")
  set(ENV{PATH} "${THEROCK_TOOLCHAIN_ROOT}/bin${PS}${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin${PS}${CURRENT_PATH}")
  message(STATUS "Augmented toolchain PATH=$ENV{PATH}")
endblock()

# Tensile is using msgpack and will pull in Boost otherwise.
add_compile_definitions(MSGPACK_NO_BOOST)

if(NOT WIN32)
  # Configure roctracer if on a supported operating system (Linux).
  # rocBLAS has deprecated dependencies on roctracer. We apply a patch to redirect
  # naked linking against `-lroctx64` to an explicitly found version of the library.
  # See: https://github.com/ROCm/TheRock/issues/364
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
  find_library(_therock_legacy_roctx64 roctx64 REQUIRED)
  cmake_language(DEFER CALL therock_patch_linked_lib OLD_LIBRARY "roctx64" NEW_TARGET "${_therock_legacy_roctx64}")
endif()

# Tensile's source-kernel compilation (which produces Kernels.so-000-{arch}.hsaco)
# requires explicit xnack qualifiers for CDNA targets that support xnack variants.
# Without them, Tensile compiles bare gfx942/gfx90a images that the HIP runtime
# rejects with hipErrorInvalidImage on hardware that reports xnack feature flags
# (e.g. MI300X reports gfx942:sramecc+:xnack-). Both xnack- and xnack+ variants
# are needed: production CDNA systems run xnack- by default; some deployments and
# ROCm tools (e.g. rocr page migration) require xnack+.
# TheRock registers bare gfx942/gfx90a/gfx950 targets; we expand each bare CDNA
# target into its two xnack variants here, matching the legacy build-infra behaviour.
# ASAN builds are excluded: therock_sanitizers.cmake already forces xnack+ only.
if(NOT THEROCK_SANITIZER)
  set(_xnack_arches "gfx90a" "gfx942" "gfx950")
  set(_tensile_xnack_targets)
  foreach(_t IN LISTS GPU_TARGETS)
    if("${_t}" IN_LIST _xnack_arches)
      list(APPEND _tensile_xnack_targets "${_t}:xnack-" "${_t}:xnack+")
    else()
      list(APPEND _tensile_xnack_targets "${_t}")
    endif()
  endforeach()
  if(NOT "${_tensile_xnack_targets}" STREQUAL "${GPU_TARGETS}")
    message(STATUS "rocBLAS: expanding Tensile GPU targets with xnack variants: ${_tensile_xnack_targets}")
    set(GPU_TARGETS "${_tensile_xnack_targets}" CACHE STRING "GPU targets" FORCE)
    set(AMDGPU_TARGETS "${_tensile_xnack_targets}" CACHE STRING "AMDGPU targets" FORCE)
  endif()
endif()
