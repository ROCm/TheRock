# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Post-install hook for FFTW3 to make the CMake config relocatable.
# FFTW3's generated config files use CMAKE_INSTALL_FULL_* variables which
# embed absolute paths. This breaks artifact copying between build trees
# (e.g., when copying compiler-runtime artifacts in CI).
#
# This hook rewrites the config files to compute paths relative to
# CMAKE_CURRENT_LIST_DIR, making them relocatable.

message(STATUS "FFTW3 post-hook: registering install-time CMake config patching (CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})")

set(_fftw3_relocate_code [=[
  # Rewrite FFTW3Config.cmake and FFTW3LibraryDepends.cmake to be relocatable
  message(STATUS "FFTW3 post-hook: RUNNING INSTALL CODE - CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
  set(_config_dir "${CMAKE_INSTALL_PREFIX}/lib/cmake/fftw3")
  set(_config_file "${_config_dir}/FFTW3Config.cmake")
  set(_depends_file "${_config_dir}/FFTW3LibraryDepends.cmake")

  message(STATUS "FFTW3 post-hook: Looking for ${_config_file}")

  if(NOT EXISTS "${_config_file}")
    message(STATUS "FFTW3Config.cmake not found at ${_config_file}, skipping relocatable patch")
    return()
  endif()

  message(STATUS "Making FFTW3 CMake configs relocatable in: ${_config_dir}")

  # Read the current config
  file(READ "${_config_file}" _config_content)

  # Check if already patched (contains our marker)
  if(_config_content MATCHES "THEROCK_RELOCATABLE_FFTW3")
    message(STATUS "FFTW3Config.cmake already patched for relocatability")
  else()
    # Create relocatable FFTW3Config.cmake
    set(_relocatable_config [==[
# THEROCK_RELOCATABLE_FFTW3 - This file has been patched by TheRock for relocatability.
# Original hardcoded paths have been replaced with paths relative to CMAKE_CURRENT_LIST_DIR.

# defined since 2.8.3
if (CMAKE_VERSION VERSION_LESS 2.8.3)
  get_filename_component (CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
endif ()

# Allows loading FFTW3 settings from another project
set (FFTW3_CONFIG_FILE "${CMAKE_CURRENT_LIST_FILE}")

# Compute the installation prefix relative to this config file location.
# This file is at <prefix>/lib/cmake/fftw3/FFTW3Config.cmake
get_filename_component(_FFTW3_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set (FFTW3_LIBRARIES fftw3)
set (FFTW3_LIBRARY_DIRS "${_FFTW3_PREFIX}/lib")
set (FFTW3_INCLUDE_DIRS "${_FFTW3_PREFIX}/include")

include ("${CMAKE_CURRENT_LIST_DIR}/FFTW3LibraryDepends.cmake" OPTIONAL)

unset(_FFTW3_PREFIX)

if (CMAKE_VERSION VERSION_LESS 2.8.3)
  set (CMAKE_CURRENT_LIST_DIR)
endif ()
]==])

    file(WRITE "${_config_file}" "${_relocatable_config}")
    message(STATUS "FFTW3Config.cmake patched for relocatability")
  endif()

  # Now patch FFTW3LibraryDepends.cmake which contains the imported target definitions
  if(EXISTS "${_depends_file}")
    file(READ "${_depends_file}" _depends_content)

    if(_depends_content MATCHES "THEROCK_RELOCATABLE")
      message(STATUS "FFTW3LibraryDepends.cmake already patched for relocatability")
    else()
      # Create relocatable FFTW3LibraryDepends.cmake
      # This defines the FFTW3::fftw3 imported target with relocatable paths
      set(_relocatable_depends [==[
# THEROCK_RELOCATABLE - This file has been patched by TheRock for relocatability.

# Compute prefix from this file's location
get_filename_component(_FFTW3_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# Create imported target FFTW3::fftw3
if(NOT TARGET FFTW3::fftw3)
  add_library(FFTW3::fftw3 SHARED IMPORTED)
  set_target_properties(FFTW3::fftw3 PROPERTIES
    IMPORTED_LOCATION "${_FFTW3_IMPORT_PREFIX}/lib/libfftw3.so.3"
    IMPORTED_SONAME "libfftw3.so.3"
    INTERFACE_INCLUDE_DIRECTORIES "${_FFTW3_IMPORT_PREFIX}/include"
  )
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
]==])

      file(WRITE "${_depends_file}" "${_relocatable_depends}")
      message(STATUS "FFTW3LibraryDepends.cmake patched for relocatability")
    endif()
  else()
    message(STATUS "FFTW3LibraryDepends.cmake not found, skipping")
  endif()
]=])

string(CONFIGURE "${_fftw3_relocate_code}" _fftw3_relocate_code @ONLY)
install(CODE "${_fftw3_relocate_code}")
