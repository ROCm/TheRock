# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-glib-sources
      SOURCE_DIR "${_source_dir}"
      # Originally from: https://download.gnome.org/sources/glib/2.74/glib-2.74.7.tar.xz
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/glib-2.74.7.tar.xz"
      URL_HASH "SHA256=196ab86c27127a61b7a70c3ba6af7b97bdc01c07cd3b21abd5e778b955eccb1b"
      TOUCH "${_download_stamp}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-glib
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
      # glib links pcre2 (core), libffi (gobject) and zlib (gio).
      RUNTIME_DEPS
        therock-libffi
        therock-pcre2
        therock-zlib
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-glib glib lib/rocm_sysdeps/lib/cmake/glib)
    therock_cmake_subproject_activate(therock-glib)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libglib-2.0.so libgobject-2.0.so libgio-2.0.so libgmodule-2.0.so libgthread-2.0.so
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(GLIB_BUILD VERSION "2.74.7")

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
    # Use cp -a rather than `cmake -E copy_directory`: GLib's test fixtures
    # include (dangling) symlinks and special files that copy_directory rejects.
    bash -c "cp -a '${SOURCE_DIR}' '${PATCH_SOURCE_DIR}'"
  COMMAND
    "${CMAKE_COMMAND}" -E chdir "${PATCH_SOURCE_DIR}"
    "${CMAKE_COMMAND}" -E env
      # Escaping hack (from libdrm): experimentally determined to persist through
      # the layers. The version script gives glib's public symbols the shared
      # AMDROCM_SYSDEPS version; hidden-visibility internals stay local.
      "LDFLAGS=-Wl,-rpath='$$ORIGIN' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/glib.map"
      "CC=${_meson_cc}"
      "CXX=${_meson_cxx}"
      --
      "${MESON_BUILD}" setup "${CMAKE_CURRENT_BINARY_DIR}"
        --reconfigure
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        -Dtests=false
        -Dman=false
        -Dselinux=disabled
        -Dlibmount=disabled
        -Dnls=disabled
        -Dglib_debug=disabled
        -Dlibelf=disabled
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
  ${CMAKE_CURRENT_SOURCE_DIR}/glib-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/glib-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/glib-config.cmake" DESTINATION lib/cmake/glib)
