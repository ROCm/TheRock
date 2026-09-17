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

# hipBLASLt imports its sanitized _rocisa Python extension while generating
# device libraries. A RUNPATH lets the loader locate compiler-rt, but TSAN must
# be loaded before Python allocates static TLS or dlopen fails with
# "cannot allocate memory in static TLS block". Upstream's TSAN option routes
# its bundled Python/code-generation commands through an LD_PRELOAD launcher.
# It does not duplicate TheRock's sanitizer flags: the upstream target flag
# helper returns immediately whenever THEROCK_SANITIZER is set.
if(THEROCK_SANITIZER STREQUAL "HOST_TSAN")
  set(HIPBLASLT_ENABLE_TSAN ON CACHE BOOL
    "Enable hipBLASLt's TSAN-aware Python code-generation launcher" FORCE)
endif()
