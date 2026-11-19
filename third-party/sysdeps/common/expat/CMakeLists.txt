# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-expat-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: https://github.com/libexpat/libexpat/releases/download/R_2_7_5/expat-2.7.5.tar.xz
      URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/expat-2.7.5.tar.xz
      URL_HASH SHA256=1032dfef4ff17f70464827daa28369b20f6584d108bc36f17ab1676e1edd2f91
      TOUCH "${_download_stamp}"
      # We need DOWNLOAD_EXTRACT_TIMESTAMP set to TRUE to preserve the timestamp
      # of the extracted sources, otherwise the build system will notice
      # changes and will try to re-run autotools, causing issues.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-expat
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
    therock_cmake_subproject_provide_package(therock-expat expat lib/rocm_sysdeps/lib/cmake/expat)
    therock_cmake_subproject_activate(therock-expat)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libexpat.so
    )

    add_test(
      NAME expat_header_test_1
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/expat.h"
    )

    add_test(
      NAME expat_header_test_2
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/expat_config.h"
    )

    add_test(
      NAME expat_header_test_3
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/expat_external.h"
    )

   return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(EXPAT_BUILD VERSION "2.7.5")

include("${THEROCK_SOURCE_DIR}/cmake/therock_meson_env.cmake")
therock_get_meson_compiler_env(_expat_cc _unused_expat_cxx)
set(_expat_base_cc "${_expat_cc}")

# Autoconf supplies -g -O2 only when CFLAGS is absent. Keep that behavior and
# put general hardening options in the compiler command. When the user did set
# CFLAGS, use its actual optimization level to decide fortification.
# If this wrapper gains an AMD compiler toolchain, sanitizer enablement must
# also pass the sanitizer compile and link options into the Autotools build.
set(_hardening_fortify ON)
if(THEROCK_SANITIZER)
  set(_hardening_fortify OFF)
elseif(DEFINED ENV{CFLAGS})
  therock_hardening_flags_are_optimized(
    _hardening_fortify "$ENV{CFLAGS}"
  )
endif()
therock_get_hardening_command_flags(
  _hardening_c_flags
  _hardening_shared_link_flags
  _hardening_executable_c_flags
  _hardening_executable_link_flags
  LANGUAGE C
  FORTIFY "${_hardening_fortify}"
)
# Append policy options after all autotools/user arguments. In particular,
# later -Wp definitions must not override fortify enablement or its reset.
# Keeping CFLAGS untouched also preserves Autoconf's default optimization.
function(_expat_wrap_compiler out_var flags)
  if(NOT flags)
    set(${out_var} "${_expat_base_cc}" PARENT_SCOPE)
    return()
  endif()
  set(_wrapper "${CMAKE_CURRENT_BINARY_DIR}/${out_var}.sh")
  file(GENERATE OUTPUT "${_wrapper}"
    CONTENT "#!/bin/sh\nexec ${_expat_base_cc} \"$@\" ${flags}\n"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
  set(${out_var} "${out_var}.sh" PARENT_SCOPE)
endfunction()
_expat_wrap_compiler(_expat_cc "${_hardening_c_flags}")
_expat_wrap_compiler(_expat_executable_cc "${_hardening_executable_c_flags}")
# Autoconf splits CC as a command; resolve wrapper names through PATH so their
# build-directory paths can contain spaces.
set(_expat_build_env)
if(_hardening_c_flags)
  list(APPEND _expat_build_env "PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PATH}")
endif()
set(_expat_make "${CMAKE_COMMAND}" -E env ${_expat_build_env} -- make)
set(_expat_executable_ld "${_expat_base_cc}")
if(_hardening_executable_link_flags)
  string(APPEND
    _expat_executable_ld " ${_hardening_executable_link_flags}"
  )
endif()

# Preserve explicit CFLAGS across later builds without suppressing Autoconf's
# optimization defaults when the variable was originally absent.
set(_expat_configure_env)
if(DEFINED ENV{CFLAGS})
  list(APPEND _expat_configure_env "CFLAGS=$ENV{CFLAGS}")
endif()

# Apply patch source only on Linux
set(patch_source_commands)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND patch_source_commands   COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${SOURCE_DIR}")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  #
  # OPTIONAL: Patching of the sources should happen here.
  #
  COMMAND
    ${patch_source_commands}
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
      "CC=${_expat_cc}"
      ${_expat_build_env}
      ${_expat_configure_env}
      # Escaping: Double $ to satisfy CMake, then double $ to satisfy configure,
      # then escaped single quotes to make it to the linker command line.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} ${_hardening_shared_link_flags} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
      "CPPFLAGS=${EXTRA_CPPFLAGS}"
      --
    "${SOURCE_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --libdir "${CMAKE_INSTALL_PREFIX}/lib"
      # Disable examples and tests.
      --without-examples
      --without-tests
      # Disable docbook-based man page generation. The xmlwf man page is
      # not part of TheRock's artifacts, and expat 2.7.x configure errors
      # out on SGML-only docbook2man (the default on Debian/Ubuntu via
      # docbook-utils) rather than falling back gracefully.
      --without-docbook
      # Disable maintainer-mode, otherwise our copy of the sources directory
      # will potentially alter the timestamp of acinclude.m4, which will
      # trigger a reconfigure with a version of aclocal we do not have.
      --disable-maintainer-mode
  # Autotools does not track compiler or linker command changes as build
  # dependencies. Clean after reconfiguration so policy switches rebuild all
  # objects with the current hardening flags.
  COMMAND
    ${_expat_make} clean
  COMMAND
    ${_expat_make} -C lib V=1
  COMMAND
    ${_expat_make} -C xmlwf V=1
      "CC=${_expat_executable_cc}"
      "CCLD=${_expat_executable_ld}"
  COMMAND
    ${_expat_make} install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/expat-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/expat-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/expat-config.cmake" DESTINATION lib/cmake/expat)
