# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libnl-sources
      SOURCE_DIR "${_source_dir}"
      # Originally from: https://github.com/thom311/libnl/releases/download/libnl3_12_0/libnl-3.12.0.tar.gz
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libnl-3.12.0.tar.gz"
      URL_HASH "SHA256=fc51ca7196f1a3f5fdf6ffd3864b50f4f9c02333be28be4eeca057e103c0dd18"

      TOUCH "${_download_stamp}"
      # We need DOWNLOAD_EXTRACT_TIMESTAMP set to TRUE to preserve the timestamp
      # of the extracted sources, otherwise the build system will notice
      # changes and will try to re-run autotools, causing issues.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-libnl
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
    therock_cmake_subproject_provide_package(therock-libnl libnl lib/rocm_sysdeps/lib/cmake/libnl)
    therock_cmake_subproject_activate(therock-libnl)

    # Validate the multiple libraries produced by libnl
    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libnl-3.so libnl-genl-3.so libnl-route-3.so libnl-idiag-3.so libnl-nf-3.so libnl-xfrm-3.so
    )

    # Test key headers
    set(LIBNL_HEADERS "netlink/netlink.h" "netlink/genl/genl.h" "netlink/route/route.h")
    foreach(libnl_header ${LIBNL_HEADERS})
      string(REPLACE "/" "_" header_test_name "${libnl_header}")
      add_test(
        NAME libnl_header_test:${header_test_name}
        COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/${libnl_header}"
      )
    endforeach()

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBNL_BUILD VERSION "3.12.0")

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
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\' "
      --
    "${SOURCE_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --disable-cli  # Disable CLI utilities library for minimal build
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
  ${CMAKE_CURRENT_SOURCE_DIR}/libnl-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libnl-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libnl-config.cmake" DESTINATION lib/cmake/libnl)
