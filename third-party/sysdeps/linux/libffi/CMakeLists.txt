# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Fetch sources
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libffi-sources
      SOURCE_DIR "${_source_dir}"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libffi-3.5.2.tar.gz"
      URL_HASH "SHA256=f3a3082a23b37c293a4fcd1053147b371f2ff91fa7ea1b2a52e335676bac82dc"
      TOUCH "${_download_stamp}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-libffi
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
      INSTALL_DESTINATION lib/rocm_sysdeps
      INTERFACE_LINK_DIRS lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS "${_download_stamp}"
    )

    therock_cmake_subproject_provide_package(therock-libffi FFI lib/rocm_sysdeps/lib/cmake/ffi)
    therock_cmake_subproject_activate(therock-libffi)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libffi.so
    )

    return()
endif()

cmake_minimum_required(VERSION 3.25)
project(LIBFFI_BUILD VERSION "3.5.2")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

set(patch_source_commands)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND patch_source_commands COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${SOURCE_DIR}")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND ${patch_source_commands}
  COMMAND
    "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/version.lds" "${CMAKE_CURRENT_BINARY_DIR}/version.lds"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\'"
      "CFLAGS=-fPIC ${CMAKE_C_FLAGS}"
      --
    "${SOURCE_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --disable-static
      --enable-shared
  COMMAND make V=1
  COMMAND make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ffi-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ffi-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ffi-config.cmake" DESTINATION lib/cmake/ffi)
