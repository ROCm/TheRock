# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message(STATUS "Customizing zstd options for TheRock")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_LIBDIR "lib")  # No lib64 for us, thank you very much.

# On Windows, build the shared library with a private DLL name so that a user
# whose PATH includes our bin/ cannot get our zstd confused with any other
# zstd.dll on the system.  This is the Windows equivalent of the private SONAME
# used on Linux (there the real object is librocm_sysdeps_zstd.so.N and only the
# linker symlink is the plain libzstd.so).  Windows has no SONAME, so the DLL
# *filename* is the runtime identity and must carry the private name.  The
# import library keeps the conventional "zstd" base name so consumers still link
# with -lzstd (mirroring Linux, where the link name stays plain).
#
# The zstd shared target (libzstd_shared) is defined in a subdirectory added
# after this file runs, so set these properties with a deferred call.  Defer on
# the current (top-level) directory -- it finishes processing after all
# subdirectories, by which point the target exists (targets are visible across
# directory scopes).  Deferring onto the lib subdirectory directly does not
# work: that directory has not been added to the build yet at this point.
if(WIN32)
  cmake_language(DEFER CALL
    set_target_properties libzstd_shared PROPERTIES
      OUTPUT_NAME "rocm_sysdeps_zstd"
      ARCHIVE_OUTPUT_NAME "zstd")
endif()
