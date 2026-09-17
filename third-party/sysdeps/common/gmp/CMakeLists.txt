# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-gmp-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/gmp-6.3.0.tar.xz"
      URL_HASH "SHA512=e85a0dab5195889948a3462189f0e0598d331d3457612e2d3350799dba2e244316d256f8161df5219538eb003e4b5343f989aaa00f96321559063ed8c8f29fd2"
      PATCH_COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_tools/patch_third_party_source.py ${CMAKE_SOURCE_DIR}/patches/third-party/gmp

      TOUCH "${_download_stamp}"
      # We need DOWNLOAD_EXTRACT_TIMESTAMP set to TRUE to preserve the timestamp
      # of the extracted sources, otherwise the build system will notice
      # changes and will try to re-run autotools, causing issues.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # On Windows, an autotools sysdep is built by the MSYS2 ucrt64 gcc, not by
    # the ambient MSVC, so name that toolchain: without it CMAKE_C_COMPILER is
    # cl.exe, and the build would depend on the login shell's PATH to supply a
    # usable compiler.
    set(_sysdep_toolchain)
    if(WIN32)
      set(_sysdep_toolchain host-gnu)
    endif()
    therock_autotools_subproject_args(therock-gmp _autotools_args)

    therock_cmake_subproject_declare(therock-gmp
      HARDENING_DEFERRED "gmp (autotools)"
      ${_autotools_args}
      COMPILER_TOOLCHAIN ${_sysdep_toolchain}
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
        "-DPATCHELF=${PATCHELF}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
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
    therock_cmake_subproject_provide_package(therock-gmp gmp lib/rocm_sysdeps/lib/cmake/gmp)
    therock_cmake_subproject_activate(therock-gmp)

    # On Windows the library is built static (libgmp.a), so there is no shared
    # object to validate.
    if(NOT WIN32)
      therock_test_validate_shared_lib(
        PATH build/dist/lib/rocm_sysdeps/lib
        LIB_NAMES libgmp.so
      )
    endif()

    # Checked through cmake instead of ls, which a ctest run from cmd.exe or
    # PowerShell on Windows does not have.  md5sum fails on a missing file.
    add_test(
      NAME gmp_header_test
      COMMAND "${CMAKE_COMMAND}" -E md5sum "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/gmp.h"
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(GMP_BUILD VERSION "6.3.0")

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

include("${THEROCK_SOURCE_DIR}/cmake/therock_autotools.cmake")

# What differs per platform is the environment configure runs with, the
# arguments it is given, and the steps around it.  How those get run is
# therock_autotools_shell()'s business.
set(configure_env)
set(pre_commands)
set(post_commands)

# Arguments both platforms configure with.  The branch below adds only what
# differs.
set(configure_args
  "--prefix=${CMAKE_INSTALL_PREFIX}"
  "--libdir=${CMAKE_INSTALL_PREFIX}/lib")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # GMP builds static here, for rocgdb to link.
  list(APPEND configure_args
    --enable-static --disable-shared)

  list(APPEND post_commands
    COMMAND
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX})

else()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT PATCHELF)
    message(FATAL_ERROR "Missing PATCHELF from super-project")
  endif()

  set(configure_env
    "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
    # Escaping: Double $ to satisfy CMake, then double $ to satisfy configure,
    # then escaped single quotes to make it to the linker command line.
    #
    # '$$ORIGIN' does make its way to LDFLAGS correctly, but doesn't survive
    # making it to the shared library's RUNPATH.
    "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
    "CPPFLAGS=${EXTRA_CPPFLAGS}")

  # Apply patch source only on Linux
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND pre_commands   COMMAND
      bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${SOURCE_DIR}")
  endif()

  list(APPEND post_commands
    COMMAND
      "${CMAKE_COMMAND}" -E env
        "PATCHELF=${PATCHELF}"
        "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
        "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX})
endif()

therock_autotools_shell(
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SOURCE_DIR "${SOURCE_DIR}"
  CONFIGURE_ENV ${configure_env}
  RUN_VAR run
  RUN_CONFIGURE_VAR run_configure
  CONFIGURE_ARGS_VAR host_args
  CONFIGURE_VAR configure)

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  #
  # OPTIONAL: Patching of the sources should happen here.
  #
  ${pre_commands}
  COMMAND ${run_configure} ${configure} ${host_args} ${configure_args}
  COMMAND ${run} make -j "${PAR_JOBS}" V=1
  COMMAND ${run} make install
  ${post_commands}
)

# The package config that find_package(gmp) consumes is needed on every
# platform, so both branches above fall through to here.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/gmp-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/gmp-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gmp-config.cmake" DESTINATION lib/cmake/gmp)
