# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libcurl-sources
      SOURCE_DIR "${_source_dir}"
      # Originally from: https://curl.se/download/curl-8.5.0.tar.xz
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/curl-8.5.0.tar.xz"
      URL_HASH "SHA256=42ab8db9e20d8290a3b633e7fbb3cec15db34df65fd1015ef8ac1e4723750eeb"
      TOUCH "${_download_stamp}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-libcurl
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
      INTERFACE_INCLUDE_DIRS
        lib/rocm_sysdeps/include
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-libcurl CURL lib/rocm_sysdeps/lib/cmake/libcurl)
    therock_cmake_subproject_activate(therock-libcurl)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libcurl.so
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBCURL_BUILD VERSION "8.5.0")

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
      # curl controls its own symbol exports via libtool + hidden visibility,
      # which conflicts with an added version script, so rely on the SONAME
      # rename alone for coexistence. TLS is intentionally omitted: amd-smi only
      # uses libfwupd's local (D-Bus) BIOS-setting path.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\'"
      --
    "${SOURCE_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --libdir "${CMAKE_INSTALL_PREFIX}/lib"
      --disable-static
      --without-ssl
      --without-libpsl
      --disable-ldap
      --disable-manual
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
  ${CMAKE_CURRENT_SOURCE_DIR}/libcurl-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libcurl-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libcurl-config.cmake" DESTINATION lib/cmake/libcurl)
