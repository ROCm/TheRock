# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libiconv-sources
      SOURCE_DIR "${_source_dir}"
      # TODO: mirror to rocm-third-party-deps.s3 like the other sysdeps.
      URL "https://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.19.1.tar.gz"
      URL_HASH "SHA512=6e737df70a7f75791bb2425078488a2a630d9deb705821326e106b935114d750ba3cb14e294e29a76c306fb3b5424ae41f4adf63060674130e67bc01b12be531"
      TOUCH "${_download_stamp}"
      # Preserve the extracted timestamps, otherwise the build system notices
      # changes and tries to re-run autotools.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # libiconv is a Windows-only sysdep, built by the MSYS2 ucrt64 gcc instead
    # of the ambient MSVC, so name that toolchain: without it CMAKE_C_COMPILER
    # is cl.exe, and the build would depend on the login shell's PATH to supply
    # a usable compiler.
    therock_autotools_subproject_args(therock-libiconv _autotools_args)

    therock_cmake_subproject_declare(therock-libiconv
      HARDENING_DEFERRED "libiconv (autotools)"
      ${_autotools_args}
      COMPILER_TOOLCHAIN host-gnu
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
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-libiconv iconv lib/rocm_sysdeps/lib/cmake/iconv)
    therock_cmake_subproject_activate(therock-libiconv)

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBICONV_BUILD VERSION "1.19.1")

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

# libiconv is a Windows-only sysdep: Linux and macOS have iconv in libc, so
# rocgdb only needs a bundled iconv on Windows (where the toolchain has none).
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(FATAL_ERROR "libiconv sysdep is Windows-only (Linux and macOS have iconv in libc)")
endif()

include("${THEROCK_SOURCE_DIR}/cmake/therock_autotools.cmake")

# libiconv is built static, for rocgdb to link.  The extra encodings include the
# EBCDIC code pages, which gdb's "set target-charset" offers on Linux through
# glibc's iconv.
set(configure_args
  "--prefix=${CMAKE_INSTALL_PREFIX}"
  "--libdir=${CMAKE_INSTALL_PREFIX}/lib"
  --enable-static --disable-shared
  --enable-extra-encodings)

therock_autotools_shell(
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SOURCE_DIR "${SOURCE_DIR}"
  RUN_VAR run
  RUN_CONFIGURE_VAR run_configure
  CONFIGURE_ARGS_VAR host_args
  CONFIGURE_VAR configure)

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND ${run_configure} ${configure} ${host_args} ${configure_args}
  COMMAND ${run} make -j "${PAR_JOBS}" V=1
  COMMAND ${run} make install
  # The libtool descriptors hold the build tree's absolute paths, and nothing
  # that ships needs them.  The other autotools sysdeps drop theirs too.
  COMMAND "${CMAKE_COMMAND}" -E rm -f
    "${CMAKE_INSTALL_PREFIX}/lib/libiconv.la"
    "${CMAKE_INSTALL_PREFIX}/lib/libcharset.la"
)

# The package config that find_package(iconv) consumes.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/iconv-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/iconv-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/iconv-config.cmake" DESTINATION lib/cmake/iconv)
