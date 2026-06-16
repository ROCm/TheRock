# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# libatomic is part of the compiler runtime (shipped with GCC), not an upstream
# project we fetch and build. A few ROCm libraries (e.g.
# librocprofiler-sdk-rocattach, libroctracer_tool) emit out-of-line __atomic_*
# calls that cannot be lowered inline on x86-64, leaving a genuine
# `NEEDED libatomic.so.1`. Clean distros that do not ship libatomic (RHEL, SLES,
# minimal images) then fail to load them. Bundle the
# toolchain's libatomic.so.1 into lib/rocm_sysdeps so the existing
# origin-relative RPATH resolves it without an extra system dependency.
#
# libatomic is licensed GPLv3 with the GCC Runtime Library Exception 3.1.
# Redistribution of the unmodified runtime built by an Eligible Compilation
# Process (our gcc-toolset-13 build container) is permitted under that
# Exception. The license texts (COPYING.RUNTIME, COPYING3) are shipped next to
# the library and the corresponding source is mirrored (see README.md).

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

# We redistribute the runtime under the GCC Runtime Library Exception, which
# only applies when the library was produced by an Eligible Compilation Process
# (i.e. built by GCC). Guard against silently bundling a libatomic that came
# from some other toolchain, which would leave the redistribution terms
# ambiguous. GCC identifies itself via __GNUC__ without __clang__.
execute_process(
  COMMAND "${CMAKE_C_COMPILER}" -dM -E -x c /dev/null
  OUTPUT_VARIABLE _compiler_defines
  ERROR_QUIET
)
if(NOT _compiler_defines MATCHES "__GNUC__" OR _compiler_defines MATCHES "__clang__")
  message(FATAL_ERROR
    "libatomic sysdep must be sourced from GCC to satisfy the GCC Runtime "
    "Library Exception, but ${CMAKE_C_COMPILER} is not GCC. Refusing to bundle.")
endif()

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

# Ship the GPLv3 + Runtime Library Exception texts and the source-availability
# notice alongside the binary, per the redistribution terms (GPLv3 sec. 6).
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/COPYING.RUNTIME"
    "${CMAKE_CURRENT_SOURCE_DIR}/COPYING3"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
  DESTINATION share/doc/libatomic)
