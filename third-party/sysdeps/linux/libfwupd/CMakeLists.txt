# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libfwupd-sources
      SOURCE_DIR "${_source_dir}"
      # Originally from: https://github.com/fwupd/fwupd/archive/refs/tags/2.1.7.tar.gz
      # 2.1.x is the last pre-Rust release series (Rust/FFI lands in 2.2.x/main)
      # and no longer depends on libjcat or json-glib.
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/fwupd-2.1.7.tar.gz"
      URL_HASH "SHA256=338ea5b139f9a37447ff2215d0a8d2cf23851fb5a39fba7b1b0ded52a40240cc"
      TOUCH "${_download_stamp}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-libfwupd
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
        "-DMESON_BUILD=${MESON_BUILD}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_INCLUDE_DIRS
        lib/rocm_sysdeps/include
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      # libfwupd links GLib (gio/gio-unix/glib) and libcurl.
      RUNTIME_DEPS
        therock-glib
        therock-libcurl
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-libfwupd libfwupd lib/rocm_sysdeps/lib/cmake/libfwupd)
    therock_cmake_subproject_activate(therock-libfwupd)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libfwupd.so
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBFWUPD_BUILD VERSION "2.1.7")

if(NOT PATCHELF OR NOT MESON_BUILD)
  message(FATAL_ERROR "Missing PATCHELF or MESON_BUILD from super-project")
endif()

include("${THEROCK_SOURCE_DIR}/cmake/therock_meson_env.cmake")
therock_get_meson_compiler_env(_meson_cc _meson_cxx)

# Meson refuses to build if the source dir is a subdir of the build dir, so
# make it a sibling.
set(PATCH_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/../patch_source")

add_custom_target(
  meson_build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${PATCH_SOURCE_DIR}"
  COMMAND
    # cp -a: the fwupd tree contains symlinks/special files under tests/.
    bash -c "cp -a '${SOURCE_DIR}' '${PATCH_SOURCE_DIR}'"
  COMMAND
    "${CMAKE_COMMAND}" -E chdir "${PATCH_SOURCE_DIR}"
    "${CMAKE_COMMAND}" -E env
      # libfwupd already versions its symbols (LIBFWUPD_*), so we do not add a
      # second version script; the SONAME rename handles coexistence.
      "LDFLAGS=-Wl,-rpath='$$ORIGIN'"
      "CC=${_meson_cc}"
      "CXX=${_meson_cxx}"
      --
      "${MESON_BUILD}" setup "${CMAKE_CURRENT_BINARY_DIR}"
        --reconfigure
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        # library-only build: no daemon, plugins, tools, or Rust.
        -Dbuild=library
        -Ddocs=disabled
        -Dman=false
        -Dtests=false
        -Dintrospection=disabled
        -Dpolkit=disabled
        -Dsystemd=disabled
        -Dbash_completion=false
  COMMAND
    "${MESON_BUILD}" compile --verbose
  COMMAND
    "${CMAKE_COMMAND}" -E env "DESTDIR=${CMAKE_INSTALL_PREFIX}" --
      "${MESON_BUILD}" install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libfwupd-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libfwupd-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libfwupd-config.cmake" DESTINATION lib/cmake/libfwupd)
