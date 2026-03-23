# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Assemble CC/CXX env strings for Meson from CMake's compiler and launcher
# variables (which are already set correctly in the sub-project toolchain).
#
# CMAKE_C_COMPILER_LAUNCHER is a CMake list (semicolon-separated) to support
# chained launchers (e.g. "ccache;distcc"). Meson expects a space-separated
# string, so the launcher is joined with spaces before being prepended to the
# compiler path.
#
# Usage:
#   include("${THEROCK_SOURCE_DIR}/cmake/therock_meson_env.cmake")
#   therock_get_meson_compiler_env(_meson_cc _meson_cxx)
#   # Then pass to meson via cmake -E env:
#   #   "CC=${_meson_cc}"
#   #   "CXX=${_meson_cxx}"
function(therock_get_meson_compiler_env out_cc out_cxx)
  set(_cc "${CMAKE_C_COMPILER}")
  set(_cxx "${CMAKE_CXX_COMPILER}")
  if(CMAKE_C_COMPILER_LAUNCHER)
    string(REPLACE ";" " " _c_launcher "${CMAKE_C_COMPILER_LAUNCHER}")
    set(_cc "${_c_launcher} ${_cc}")
  endif()
  if(CMAKE_CXX_COMPILER_LAUNCHER)
    string(REPLACE ";" " " _cxx_launcher "${CMAKE_CXX_COMPILER_LAUNCHER}")
    set(_cxx "${_cxx_launcher} ${_cxx}")
  endif()
  set(${out_cc} "${_cc}" PARENT_SCOPE)
  set(${out_cxx} "${_cxx}" PARENT_SCOPE)
endfunction()
