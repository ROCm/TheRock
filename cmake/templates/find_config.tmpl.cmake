# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Template find_package config file used to trampoline to a fixed location.
# Processed via configure_file with the following variables in scope:
#   @_package_name@ : Name of the package being searched
#   @_package_name_lower@ : Lowercase name of the package being searched
#   @_find_package_relpath@ : Relative path from this file's directory to the
#                             real config directory (relocatable)

# This trampolines through the list of file patterns for config scripts that
# is prescribed in the find_package docs. Since this is presumed to be for
# a component of the super project that must exist, it is a fatal error if a
# suitable destination is not found.

# Compute the absolute path at include-time using CMAKE_CURRENT_LIST_DIR.
# This makes the trampoline relocatable - it works regardless of where the
# build tree is moved (e.g., when copying artifacts between CI runs).
set(_find_package_relpath "@_find_package_relpath@")
cmake_path(ABSOLUTE_PATH _find_package_relpath
  BASE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  NORMALIZE
  OUTPUT_VARIABLE _therock_find_package_path)

if(EXISTS "${_therock_find_package_path}/@_package_name@Config.cmake")
  message(STATUS "Super-project find_package(${CMAKE_FIND_PACKAGE_NAME}) -> ${_therock_find_package_path}/@_package_name@Config.cmake")
  include("${_therock_find_package_path}/@_package_name@Config.cmake")
elseif(EXISTS "${_therock_find_package_path}/@_package_name_lower@-config.cmake")
  message(STATUS "Super-project find_package(${CMAKE_FIND_PACKAGE_NAME}) -> ${_therock_find_package_path}/@_package_name_lower@-config.cmake")
  include("${_therock_find_package_path}/@_package_name_lower@-config.cmake")
else()
  message(FATAL_ERROR "Super-project based find_package(@_package_name@) config "
    "file not found under ${_therock_find_package_path} (relative: @_find_package_relpath@)"
  )
endif()

unset(_therock_find_package_path)
unset(_find_package_relpath)
