# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libmnl-sources
      SOURCE_DIR "${_source_dir}"
      # Originally from: https://netfilter.org/pub/libmnl/libmnl-1.0.5.tar.bz2
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libmnl-1.0.5.tar.bz2"
      URL_HASH "SHA256=274b9b919ef3152bfb3da3a13c950dd60d6e2bcd54230ffeca298d03b40d0525"

      TOUCH "${_download_stamp}"
      # We need DOWNLOAD_EXTRACT_TIMESTAMP set to TRUE to preserve the timestamp
      # of the extracted sources, otherwise the build system will notice
      # changes and will try to re-run autotools, causing issues.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-libmnl
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
    therock_cmake_subproject_provide_package(therock-libmnl libmnl lib/rocm_sysdeps/lib/cmake/libmnl)
    therock_cmake_subproject_activate(therock-libmnl)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libmnl.so
    )

    add_test(
      NAME libmnl_header_test
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/libmnl/libmnl.h"
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBMNL_BUILD VERSION "1.0.5")

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

set(patch_source_commands)
list(APPEND patch_source_commands   COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${SOURCE_DIR}")

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"

  COMMAND
    ${patch_source_commands}
  COMMAND
    "${CMAKE_COMMAND}" -E env
      # Escaping: Double $ to satisfy CMake, then double $ to satisfy configure,
      # then escaped single quotes to make it to the linker command line.
      #
      # '$$ORIGIN' does make its way to LDFLAGS correctly, but doesn't survive
      # making it to the shared library's RUNPATH.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\'"
      --
    "${SOURCE_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
  COMMAND
    make V=1
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libmnl-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libmnl-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libmnl-config.cmake" DESTINATION lib/cmake/libmnl)
