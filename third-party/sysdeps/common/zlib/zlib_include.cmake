# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message(STATUS "Customizing zlib options for TheRock")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# On Windows, build the shared library with a private DLL name so that a user
# whose PATH includes our bin/ cannot get our zlib confused with any other
# z.dll/zlib1.dll on the system.  This is the Windows counterpart of the private
# SONAME used on Linux (real object librocm_sysdeps_z.so.N, plain linker symlink
# libz.so).  Windows has no SONAME, so the DLL filename is the runtime identity
# and carries the private name.  The import library keeps the conventional "z"
# base name so consumers still link with -lz.  zlib's shared target is named
# "zlib" (with OUTPUT_NAME z).  The static target is "zlibstatic" and is left
# alone.  The target does not exist yet when this file runs
# (CMAKE_PROJECT_TOP_LEVEL_INCLUDES is injected before add_library), so defer
# the call to the end of the current directory.
if(WIN32)
  cmake_language(DEFER CALL
    set_target_properties zlib PROPERTIES
      OUTPUT_NAME "rocm_sysdeps_z"
      ARCHIVE_OUTPUT_NAME "z")
endif()
