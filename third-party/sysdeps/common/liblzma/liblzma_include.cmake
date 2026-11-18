# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message(STATUS "Customizing xz/liblzma options for TheRock")
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
set(CMAKE_INSTALL_LIBDIR "lib")  # No lib64 for us, thank you very much.

# Build the shared library on Linux and Windows.  On Linux the private name and
# symbol versioning are applied via patch_source.sh + version.lds.  The Windows
# naming is handled below.
#
# xz builds one library type per configuration, so on Windows the recipe
# configures a second build for the static library, passing
# -DTHEROCK_LIBLZMA_STATIC=ON.
if(THEROCK_LIBLZMA_STATIC)
  set(BUILD_SHARED_LIBS OFF)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(BUILD_SHARED_LIBS ON)
endif()

# On Windows, give the DLL a private name so that a user whose PATH includes our
# bin/ cannot get our liblzma confused with any other liblzma DLL on the system.
# This is the Windows counterpart of the private SONAME used on Linux (real
# object librocm_sysdeps_liblzma.so.N, plain linker symlink liblzma.so).
# Windows has no SONAME, so the DLL filename is the runtime identity and carries
# the private name.  The import library keeps the conventional "lzma" base name
# so consumers still link with -llzma.
#
# xz's liblzma target sets OUTPUT_NAME "lzma", and for a shared build on
# non-mingw Windows it overrides the DLL's RUNTIME_OUTPUT_NAME to "liblzma".
# The target does not exist yet when this file runs
# (CMAKE_PROJECT_TOP_LEVEL_INCLUDES is injected at project(), before
# add_library), so defer the property change to the end of the current
# directory.
#
# The static library is named liblzmastatic.lib.  xz would name it lzma.lib, the
# same as the DLL's import library, and the two go to the same lib/.
if(WIN32 AND THEROCK_LIBLZMA_STATIC)
  cmake_language(DEFER CALL
    set_target_properties liblzma PROPERTIES
      ARCHIVE_OUTPUT_NAME "liblzmastatic")
elseif(WIN32)
  cmake_language(DEFER CALL
    set_target_properties liblzma PROPERTIES
      RUNTIME_OUTPUT_NAME "rocm_sysdeps_liblzma"
      ARCHIVE_OUTPUT_NAME "lzma")
endif()
