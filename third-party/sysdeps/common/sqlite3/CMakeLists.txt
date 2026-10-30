# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(SQLITE3_VERSION "3.51.3")
    # SQLite encodes the version in amalgamation download URLs as MMNNPP00,
    # with zero-padded minor/patch (e.g. 3.51.3 -> 3510300).
    set(SQLITE3_AMALGAMATION_VERSION "3510300")

    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-sqlite3-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://sqlite.org/2026/sqlite-amalgamation-${SQLITE3_AMALGAMATION_VERSION}.zip"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/sqlite-amalgamation-${SQLITE3_AMALGAMATION_VERSION}.zip"
      URL_HASH "SHA256=acb1e6f5d832484bf6d32b681e858c38add8b2acdfd42ac5df24b8afb46552b4"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-sqlite3
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR .
      FPRINT_SOURCE_DIR "${_source_dir}"
      FPRINT_FILE_GLOBS "${CMAKE_CURRENT_LIST_DIR}/*"
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DSQLITE3_VERSION=${SQLITE3_VERSION}"
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-sqlite3 SQLite3 lib/rocm_sysdeps/lib/cmake/SQLite3)
    therock_cmake_subproject_activate(therock-sqlite3)
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(SQLITE3_BUILD)
include(CMakeDependentOption)

# SQLite is built as a static library with hidden symbols, so each ROCm
# component that uses it (MIOpen, the rocprofiler-sdk tool, profiler-hub) embeds
# a private copy, the same way rocFFT already embeds its own SQLite. A shared
# librocm_sysdeps_sqlite3 is preloaded RTLD_GLOBAL with the rest of ROCm and
# would capture the sqlite3_* references of every other SQLite user in the
# process, e.g. Python's sqlite3 module or GDAL; see
# https://github.com/ROCm/TheRock/issues/9385.
#
# SQLITE_OMIT_LOAD_EXTENSION (which rocFFT sets) is not used: MIOpen's
# MIOPEN_EMBED_DB build calls sqlite3_enable_load_extension.
set(BUILD_SHARED_LIBS OFF)

find_package(Threads REQUIRED)

add_library(
  rocm_sysdeps_sqlite3
  ${SOURCE_DIR}/sqlite3.c
)
set_target_properties(rocm_sysdeps_sqlite3 PROPERTIES
  C_VISIBILITY_PRESET "hidden"
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(rocm_sysdeps_sqlite3 PRIVATE
  ${CMAKE_DL_LIBS}
  Threads::Threads
)

# Consumers of the static library link its system dependencies themselves.
set(SQLITE3_STATIC_LINK_LIBRARIES "")
if(NOT WIN32)
    # Unquoted so that empty entries (e.g. pthread on glibc >= 2.34) are dropped.
    set(SQLITE3_STATIC_LINK_LIBRARIES ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} m)
endif()

install(TARGETS rocm_sysdeps_sqlite3)
install(FILES "${SOURCE_DIR}/sqlite3.h" TYPE INCLUDE)
install(FILES "${SOURCE_DIR}/sqlite3ext.h" TYPE INCLUDE)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/sqlite3-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/sqlite3-config.cmake
  @ONLY
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/sqlite3-config-version.cmake"
  VERSION "${SQLITE3_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sqlite3-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sqlite3-config-version.cmake"
  DESTINATION lib/cmake/SQLite3
)
