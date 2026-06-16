# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# libatomic is part of the compiler runtime (shipped with GCC), not an upstream
# project we fetch and build. A few ROCm libraries (e.g.
# librocprofiler-sdk-rocattach, libroctracer_tool) emit out-of-line __atomic_*
# calls that cannot be lowered inline on x86-64, leaving a genuine
# `NEEDED libatomic.so.1`. Clean distros that do not ship libatomic (RHEL, SLES,
# minimal images) then fail to load them (ROCm/TheRock#5819). Bundle the
# toolchain's libatomic.so.1 into lib/rocm_sysdeps so the existing
# origin-relative RPATH resolves it without an extra system dependency.
#
# Redistribution is permitted under the GCC Runtime Library Exception.

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, set up the sub-project. Unlike other sysdeps we
    # do not fetch sources; the prebuilt runtime is sourced from the compiler.
    therock_cmake_subproject_declare(therock-libatomic
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR .
      FPRINT_FILE_GLOBS "${CMAKE_CURRENT_LIST_DIR}/*"
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
    )
    therock_cmake_subproject_activate(therock-libatomic)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libatomic.so.1
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBATOMIC_BUNDLE C)

# Ask the C compiler where its libatomic.so.1 lives. This resolves to the
# toolchain copy (e.g. gcc-toolset's libatomic) that is ABI-compatible with the
# rest of what we build. -print-file-name returns the input unchanged if absent.
execute_process(
  COMMAND "${CMAKE_C_COMPILER}" -print-file-name=libatomic.so.1
  OUTPUT_VARIABLE _libatomic_path
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _libatomic_path OR _libatomic_path STREQUAL "libatomic.so.1" OR NOT EXISTS "${_libatomic_path}")
  message(FATAL_ERROR "Could not locate libatomic.so.1 from compiler ${CMAKE_C_COMPILER} (got '${_libatomic_path}')")
endif()

# Resolve the symlink to the real shared object. Its SONAME is already
# libatomic.so.1, so install it under that name to match the consumer NEEDED.
file(REAL_PATH "${_libatomic_path}" _libatomic_real)
message(STATUS "Bundling libatomic from: ${_libatomic_real}")

install(FILES "${_libatomic_real}" DESTINATION lib RENAME libatomic.so.1)
