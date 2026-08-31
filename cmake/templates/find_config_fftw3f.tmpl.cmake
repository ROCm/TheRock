# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Custom relocatable find_package config for FFTW3f (float precision).
# This template defines the FFTW3f targets directly without relying on
# upstream's FFTW3fConfig.cmake, which embeds absolute paths that break
# when artifacts are copied between build trees.
#
# Processed via configure_file with:
#   @_package_name@ : Name of the package (FFTW3f)
#   @_find_package_path@ : Absolute path to the package cmake directory

# Compute prefix from this config file's location
# This file is at <prefix>/FFTW3fConfig.cmake (in the trampoline prefix)
# The actual libraries are at @_find_package_path@/../..
get_filename_component(_FFTW3F_IMPORT_PREFIX "@_find_package_path@/../.." ABSOLUTE)

message(STATUS "Super-project find_package(@_package_name@) -> relocatable config (prefix: ${_FFTW3F_IMPORT_PREFIX})")

# Set standard FFTW3f variables
set(FFTW3f_FOUND TRUE)
set(FFTW3f_LIBRARIES fftw3f)
set(FFTW3f_LIBRARY_DIRS "${_FFTW3F_IMPORT_PREFIX}/lib")
set(FFTW3f_INCLUDE_DIRS "${_FFTW3F_IMPORT_PREFIX}/include")

# Create imported target FFTW3::fftw3f
if(NOT TARGET FFTW3::fftw3f)
  add_library(FFTW3::fftw3f SHARED IMPORTED)
  if(EXISTS "${_FFTW3F_IMPORT_PREFIX}/lib/libfftw3f.so.3")
    set_target_properties(FFTW3::fftw3f PROPERTIES
      IMPORTED_LOCATION "${_FFTW3F_IMPORT_PREFIX}/lib/libfftw3f.so.3"
      IMPORTED_SONAME "libfftw3f.so.3"
      INTERFACE_INCLUDE_DIRECTORIES "${_FFTW3F_IMPORT_PREFIX}/include"
    )
  elseif(EXISTS "${_FFTW3F_IMPORT_PREFIX}/lib/libfftw3f.so")
    set_target_properties(FFTW3::fftw3f PROPERTIES
      IMPORTED_LOCATION "${_FFTW3F_IMPORT_PREFIX}/lib/libfftw3f.so"
      INTERFACE_INCLUDE_DIRECTORIES "${_FFTW3F_IMPORT_PREFIX}/include"
    )
  endif()
endif()

# Create imported target for threaded version if it exists
if(EXISTS "${_FFTW3F_IMPORT_PREFIX}/lib/libfftw3f_threads.so")
  if(NOT TARGET FFTW3::fftw3f_threads)
    add_library(FFTW3::fftw3f_threads SHARED IMPORTED)
    set_target_properties(FFTW3::fftw3f_threads PROPERTIES
      IMPORTED_LOCATION "${_FFTW3F_IMPORT_PREFIX}/lib/libfftw3f_threads.so"
      INTERFACE_INCLUDE_DIRECTORIES "${_FFTW3F_IMPORT_PREFIX}/include"
      INTERFACE_LINK_LIBRARIES "FFTW3::fftw3f"
    )
  endif()
endif()

unset(_FFTW3F_IMPORT_PREFIX)
