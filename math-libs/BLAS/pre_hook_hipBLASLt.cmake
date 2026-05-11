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

# hipBLASLt enables Fortran which uses system gfortran. The -shared-libsan flag
# is Clang-specific and not recognized by gfortran, so remove it from the global
# linker flags. The sanitizer flag (-fsanitize=address/thread) remains for all
# languages, and add_link_options() in therock_sanitizers.cmake already handles
# -shared-libsan correctly for C/C++ via generator expressions.
string(REPLACE " -shared-libsan" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
string(REPLACE " -shared-libsan" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" PARENT_SCOPE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" PARENT_SCOPE)
