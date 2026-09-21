# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    # Originally mirrored from: https://github.com/tukaani-project/xz/releases/download/v5.8.1/xz-5.8.1.tar.bz2
    set(_xz_url "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/xz-5.8.1.tar.bz2")
    therock_subproject_fetch(therock-liblzma-sources
      SOURCE_DIR "${_source_dir}"
      URL "${_xz_url}"
      URL_HASH SHA256=5965c692c4c8800cd4b33ce6d0f6ac9ac9d6ab227b17c512b6561bce4f08d47e
      TOUCH "${_download_stamp}"
    )

    # extract version from url
    string(REGEX MATCH "xz-([0-9]+\.[0-9]+\.[0-9]+)\.tar" _ "${_xz_url}")
    set(LIBLZMA_VERSION "${CMAKE_MATCH_1}")

    therock_cmake_subproject_declare(therock-liblzma
      HARDENING_DEFERRED "liblzma (CMake)"
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR .
      FPRINT_SOURCE_DIR "${_source_dir}"
      FPRINT_FILE_GLOBS
        "${CMAKE_CURRENT_LIST_DIR}/*"
        ${THEROCK_INSTALL_WINDOWS_DLLS_SCRIPT}
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DPATCHELF=${PATCHELF}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
        "-DLIBLZMA_VERSION=${LIBLZMA_VERSION}"
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_PROGRAM_DIRS
        lib/rocm_sysdeps/bin
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS
        "${_download_stamp}"
        ${THEROCK_INSTALL_WINDOWS_DLLS_SCRIPT}
    )
    # LibLZMA is TheRock's own config, written by this recipe, and liblzma is
    # the one xz installs.  Both files are named liblzma-config.cmake, so they
    # need directories whose names differ even ignoring case: on Windows a
    # lib/cmake/LIBLZMA would be the same directory as lib/cmake/liblzma, and
    # one file would overwrite the other.
    therock_cmake_subproject_provide_package(therock-liblzma LibLZMA lib/rocm_sysdeps/lib/cmake/LibLZMA-TheRock)
    therock_cmake_subproject_provide_package(therock-liblzma liblzma lib/rocm_sysdeps/lib/cmake/liblzma)
    therock_cmake_subproject_activate(therock-liblzma)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES liblzma.so
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBLZMA_BUILD VERSION "${LIBLZMA_VERSION}")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

# Apply patch source only on Linux
set(patch_source_commands)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND patch_source_commands   COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}/s")
endif()
# xz/liblzma provides a CMakeLists.txt, however, we have to do some post-processing
# of the libraries in order to prepare them for our use, so we invoke it as
# a sub-build. We do this uniformly across all platforms because it is easier
# to have common code paths. We write directly into the install prefix at
# build time since this build is only ever used in a one shot way.
set(XZ_CONFIGURE_ARGS
  "-G${CMAKE_GENERATOR}"
  "-S${CMAKE_CURRENT_BINARY_DIR}/s"
  "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${CMAKE_CURRENT_SOURCE_DIR}/liblzma_include.cmake"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
  # TODO: This should tunnel the CMAKE_BUILD_TYPE from TheRock
  "-DCMAKE_BUILD_TYPE=Release"
  # Disable symlink creation
  "-DCREATE_XZ_SYMLINKS=OFF"
  "-DCREATE_LZMA_SYMLINKS=OFF"
  # Disable writing pc files: we will do so ourselves.
  -DSKIP_INSTALL_FILES=ON
  # Set versioning to generic to override with a private map
  -DXZ_SYMBOL_VERSIONING=generic
)
# Driver builds link liblzma into comgr, through LLVM's xz support, and comgr
# uses the static MSVC runtime, as for zlib and zstd.
if(WIN32 AND THEROCK_FLAG_WINDOWS_DRIVER_BUILD)
  list(APPEND XZ_CONFIGURE_ARGS
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
    -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
  )
endif()

set(STATIC_BUILD_COMMANDS)
set(POST_INSTALL_COMMANDS)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # On Windows, build the static library as well as the DLL, as zstd and zlib
  # do.  The static library is what CMake consumers, built with MSVC, link
  # through the package config written below.  The DLL is for rocgdb, built with
  # mingw gcc, which cannot link an MSVC static library.  xz builds one library
  # type per configuration, so this is a second build, of the library alone,
  # copied in beside the DLL's import library.
  set(STATIC_BUILD_COMMANDS
    COMMAND
      "${CMAKE_COMMAND}" ${XZ_CONFIGURE_ARGS}
        "-B${CMAKE_CURRENT_BINARY_DIR}/b-static"
        -DTHEROCK_LIBLZMA_STATIC=ON
    COMMAND
      "${CMAKE_COMMAND}" --build "${CMAKE_CURRENT_BINARY_DIR}/b-static"
        --target liblzma
    COMMAND
      "${CMAKE_COMMAND}" -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/b-static/liblzmastatic.lib"
        "${CMAKE_INSTALL_PREFIX}/lib/"
  )
  # The DLL installs under this sub-project's private prefix, where the Windows
  # loader will not look for it.  Copy it to the stage's bin/ as well, keeping
  # the original for the tools installed beside it and for the exported CMake
  # targets that point at it.
  set(POST_INSTALL_COMMANDS
    COMMAND
      "${CMAKE_COMMAND}"
        "-DPREFIX=${CMAKE_INSTALL_PREFIX}"
        "-DSTAGE_DIR=${THEROCK_STAGE_INSTALL_ROOT}"
        -P "${THEROCK_SOURCE_DIR}/cmake/therock_install_windows_dlls.cmake"
  )
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    ${patch_source_commands}
  COMMAND
    # Provide our own version map with private symbol versions.
    "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/version.lds" "${CMAKE_CURRENT_BINARY_DIR}/s/src/liblzma/liblzma_generic.map"
  COMMAND
    "${CMAKE_COMMAND}" ${XZ_CONFIGURE_ARGS}
      "-B${CMAKE_CURRENT_BINARY_DIR}/b"
  COMMAND
    "${CMAKE_COMMAND}" --build "${CMAKE_CURRENT_BINARY_DIR}/b"
  COMMAND
    "${CMAKE_COMMAND}" --install "${CMAKE_CURRENT_BINARY_DIR}/b"
  ${STATIC_BUILD_COMMANDS}
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
  ${POST_INSTALL_COMMANDS}
)


# Output cmake and pkgconfig files. Note that the library names are different
# on windows vs linux.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LIBLZMA_TYPE "SHARED")
  set(LIBLZMA_IMPORTED_LOCATION "lib/liblzma.so")
  set(LIBLZMA_COMPILE_DEFINITIONS "")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # CMake consumers get the static library.  LZMA_API_STATIC is required with
  # it: without it, lzma.h declares every function __declspec(dllimport), and
  # the link fails on unresolved __imp_lzma_* symbols.  The DLL and its lzma.lib
  # import library are for rocgdb, which links -llzma through autotools rather
  # than through this config.
  set(LIBLZMA_TYPE "STATIC")
  set(LIBLZMA_IMPORTED_LOCATION "lib/liblzmastatic.lib")
  set(LIBLZMA_COMPILE_DEFINITIONS "LZMA_API_STATIC")
else()
  message(FATAL_ERROR "Unrecognized system type")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/liblzma-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/liblzma-config.cmake
  @ONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/liblzma.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/liblzma.pc
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/liblzma-config.cmake" DESTINATION lib/cmake/LibLZMA-TheRock)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/liblzma.pc" DESTINATION lib/pkgconfig)
