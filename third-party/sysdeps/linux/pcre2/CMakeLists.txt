# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-pcre2-sources
      SOURCE_DIR "${_source_dir}"
      # Originally from: https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.42/pcre2-10.42.tar.bz2
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/pcre2-10.42.tar.bz2"
      URL_HASH "SHA256=8d36cd8cb6ea2a4c2bb358ff6411b0c788633a2a45dabbf1aeb4b701d1b5e840"
      TOUCH "${_download_stamp}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-pcre2
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
    therock_cmake_subproject_provide_package(therock-pcre2 pcre2 lib/rocm_sysdeps/lib/cmake/pcre2)
    therock_cmake_subproject_activate(therock-pcre2)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libpcre2-8.so
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(PCRE2_BUILD VERSION "10.42")

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
      # pcre2 has no symbol version script of its own, so inject the shared
      # AMDROCM_SYSDEPS version directly via the linker.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/pcre2.map"
      --
    "${SOURCE_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --libdir "${CMAKE_INSTALL_PREFIX}/lib"
      --disable-static
      --enable-unicode
      --enable-jit
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
  ${CMAKE_CURRENT_SOURCE_DIR}/pcre2-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/pcre2-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pcre2-config.cmake" DESTINATION lib/cmake/pcre2)
