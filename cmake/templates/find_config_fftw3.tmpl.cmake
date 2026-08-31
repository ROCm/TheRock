# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Custom relocatable find_package config for FFTW3.
# This template defines the FFTW3 targets directly without relying on
# upstream's FFTW3Config.cmake, which embeds absolute paths that break
# when artifacts are copied between build trees.
#
# Processed via configure_file with:
#   @_package_name@ : Name of the package (FFTW3)
#   @_find_package_path@ : Absolute path to the package cmake directory

# Compute prefix from this config file's location
# This file is at <prefix>/FFTW3Config.cmake (in the trampoline prefix)
# The actual libraries are at @_find_package_path@/../..
get_filename_component(_FFTW3_IMPORT_PREFIX "@_find_package_path@/../.." ABSOLUTE)

message(STATUS "Super-project find_package(@_package_name@) -> relocatable config (prefix: ${_FFTW3_IMPORT_PREFIX})")

# Set standard FFTW3 variables
set(FFTW3_FOUND TRUE)
set(FFTW3_LIBRARIES fftw3)
set(FFTW3_LIBRARY_DIRS "${_FFTW3_IMPORT_PREFIX}/lib")
set(FFTW3_INCLUDE_DIRS "${_FFTW3_IMPORT_PREFIX}/include")

# Create imported target FFTW3::fftw3
if(NOT TARGET FFTW3::fftw3)
  add_library(FFTW3::fftw3 SHARED IMPORTED)
  if(EXISTS "${_FFTW3_IMPORT_PREFIX}/lib/libfftw3.so.3")
    set_target_properties(FFTW3::fftw3 PROPERTIES
      IMPORTED_LOCATION "${_FFTW3_IMPORT_PREFIX}/lib/libfftw3.so.3"
      IMPORTED_SONAME "libfftw3.so.3"
      INTERFACE_INCLUDE_DIRECTORIES "${_FFTW3_IMPORT_PREFIX}/include"
    )
  elseif(EXISTS "${_FFTW3_IMPORT_PREFIX}/lib/libfftw3.so")
    set_target_properties(FFTW3::fftw3 PROPERTIES
      IMPORTED_LOCATION "${_FFTW3_IMPORT_PREFIX}/lib/libfftw3.so"
      INTERFACE_INCLUDE_DIRECTORIES "${_FFTW3_IMPORT_PREFIX}/include"
    )
  endif()
endif()

# Create imported target for threaded version if it exists
if(EXISTS "${_FFTW3_IMPORT_PREFIX}/lib/libfftw3_threads.so")
  if(NOT TARGET FFTW3::fftw3_threads)
    add_library(FFTW3::fftw3_threads SHARED IMPORTED)
    set_target_properties(FFTW3::fftw3_threads PROPERTIES
      IMPORTED_LOCATION "${_FFTW3_IMPORT_PREFIX}/lib/libfftw3_threads.so"
      INTERFACE_INCLUDE_DIRECTORIES "${_FFTW3_IMPORT_PREFIX}/include"
      INTERFACE_LINK_LIBRARIES "FFTW3::fftw3"
    )
  endif()
endif()

unset(_FFTW3_IMPORT_PREFIX)
