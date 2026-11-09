# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libdrm-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://gitlab.freedesktop.org/mesa/libdrm/-/archive/libdrm-2.4.134/libdrm-libdrm-2.4.134.tar.bz2"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libdrm-libdrm-2.4.134.tar.bz2"
      URL_HASH "SHA256=D77E5AF336B6491F9E2427378244912EF5295C3C034AF76846C0FCFFE2EB869C"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-libdrm
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
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_activate(therock-libdrm)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libdrm.so libdrm_amdgpu.so
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBDRM_BUILD)

if(NOT PATCHELF OR NOT MESON_BUILD)
  message(FATAL_ERROR "Missing PATCHELF or MESON_BUILD from super-project")
endif()

include("${THEROCK_SOURCE_DIR}/cmake/therock_meson_env.cmake")
therock_get_meson_compiler_env(_meson_cc _meson_cxx)

# If this wrapper gains an AMD compiler toolchain, sanitizer enablement must
# also pass the sanitizer compile and link options into the Meson build.
set(_hardening_fortify ON)
if(THEROCK_SANITIZER)
  set(_hardening_fortify OFF)
endif()
therock_get_hardening_command_flags(
  _hardening_c_flags
  _hardening_c_link_flags
  _unused_c_executable_flags
  _unused_c_executable_link_flags
  LANGUAGE C
  FORTIFY "${_hardening_fortify}"
)
therock_get_hardening_command_flags(
  _hardening_cxx_flags
  _hardening_cxx_link_flags
  _unused_cxx_executable_flags
  _unused_cxx_executable_link_flags
  LANGUAGE CXX
  FORTIFY "${_hardening_fortify}"
)
# Cache initial CPPFLAGS alongside CMake's cached CFLAGS/CXXFLAGS so a later
# policy reconfigure retains them even when the environment is no longer set.
set(LIBDRM_CPPFLAGS "$ENV{CPPFLAGS}" CACHE STRING
  "Preprocessor flags for libdrm's Meson build")
set(_meson_c_args "${CMAKE_C_FLAGS} ${LIBDRM_CPPFLAGS}")
set(_meson_cxx_args "${CMAKE_CXX_FLAGS} ${LIBDRM_CPPFLAGS}")
set(_meson_c_link_args "-Wl,-rpath='$$ORIGIN'")
set(_meson_cxx_link_args "-Wl,-rpath='$$ORIGIN'")
set(_meson_pie false)
if(CMAKE_C_FLAGS)
  string(APPEND _meson_c_link_args " ${CMAKE_C_FLAGS}")
endif()
if(CMAKE_CXX_FLAGS)
  string(APPEND _meson_cxx_link_args " ${CMAKE_CXX_FLAGS}")
endif()
if(_hardening_c_flags)
  string(APPEND _meson_c_args " ${_hardening_c_flags}")
endif()
if(_hardening_cxx_flags)
  string(APPEND _meson_cxx_args " ${_hardening_cxx_flags}")
endif()
if(_hardening_c_link_flags)
  string(APPEND _meson_c_link_args " ${_hardening_c_link_flags}")
endif()
if(_hardening_cxx_link_flags)
  string(APPEND _meson_cxx_link_args " ${_hardening_cxx_link_flags}")
endif()
if(_hardening_c_flags)
  set(_meson_pie true)
endif()

# Meson refuses to build if the source dir is a subdir of the build dir, so
# make it a sibling.
set(PATCH_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/../patch_source")

add_custom_target(
  meson_build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${PATCH_SOURCE_DIR}"
  COMMAND
    # We have to patch the sources so make a fresh copy.
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${PATCH_SOURCE_DIR}"
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${PATCH_SOURCE_DIR}"
  COMMAND
    # Inline the amdgpu device table.
    "${CMAKE_CURRENT_SOURCE_DIR}/inline_amdgpu_ids.sh" "${PATCH_SOURCE_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E chdir "${PATCH_SOURCE_DIR}"
    "${CMAKE_COMMAND}" -E env
      "CC=${_meson_cc}"
      "CXX=${_meson_cxx}"
      --
      "${MESON_BUILD}" setup "${CMAKE_CURRENT_BINARY_DIR}"
        --reconfigure
        # Meson reads environment flags only on its first setup. Explicitly
        # setting these options on every reconfigure makes policy switches
        # update an existing build directory in both directions.
        "-Dc_args=${_meson_c_args}"
        "-Dcpp_args=${_meson_cxx_args}"
        "-Dc_link_args=${_meson_c_link_args}"
        "-Dcpp_link_args=${_meson_cxx_link_args}"
        "-Db_pie=${_meson_pie}"
        # We generate relocatable, arch neutral directory layouts and use
        # DESTDIR to install int ot the lib/rocm_sysdeps tree.
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        # Only enable the libraries we want.
        -Damdgpu=enabled
        -Dintel=disabled
        -Dman-pages=disabled
        -Dnouveau=disabled
        -Dradeon=disabled
        -Dvmwgfx=disabled
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
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
)
