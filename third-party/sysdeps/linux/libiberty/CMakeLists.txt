# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libiberty-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://ftpmirror.gnu.org/gnu/binutils/binutils-2.46.0.tar.gz"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/binutils-2.46.0.tar.gz"
      URL_HASH "SHA512=30d24f7461d459dcbd04e506bb74e487a92748ed411878aacf7388a79ac5b73f968569f43385c9f12c669a5c0b0295d3083a8a4f3ab1be1bb826b3a15ec1f029"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-libiberty
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
        "-DCC=${CMAKE_C_COMPILER}"
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

    therock_cmake_subproject_provide_package(therock-libiberty LibIberty lib/rocm_sysdeps/lib/cmake/libiberty)
    therock_cmake_subproject_activate(therock-libiberty)

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBIBERTY_BUILD VERSION 2.46.0)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

# -fPIC must be in CFLAGS/CXXFLAGS (compiler flags), so that libiberty object
# files are compiled as position-independent code and can be linked into shared
# libraries by downstream consumers (e.g. dyninst). Append to any existing
# flags from the environment so we do not discard flags set by the caller.
set(EXTRA_CFLAGS "-fPIC")
set(EXTRA_CXXFLAGS "-fPIC")
message(STATUS "EXTRA_CFLAGS=${EXTRA_CFLAGS}")
message(STATUS "EXTRA_CXXFLAGS=${EXTRA_CXXFLAGS}")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "CFLAGS=$ENV{CFLAGS} ${EXTRA_CFLAGS}"
      "CXXFLAGS=$ENV{CXXFLAGS} ${EXTRA_CXXFLAGS}"
      --
    "${CMAKE_CURRENT_BINARY_DIR}/s/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --enable-install-libiberty
      # --disable-shared and -fPIC together are intentional: we ask
      # autoconf not to build a shared library (we do that ourselves in
      # patch_install.py from the PIC-compiled static archive), while
      # -fPIC ensures the object files can be relinked into a shared
      # library.  This satisfies the LGPL relinking requirement without
      # relying on autoconf's shared-library support.
      --disable-shared
      --libdir="${CMAKE_INSTALL_PREFIX}/lib"
      --includedir="${CMAKE_INSTALL_PREFIX}/include"
      --disable-nls
      --disable-werror
      --disable-gprofng
      --disable-maintainer-mode
  COMMAND
    make -j "${PAR_JOBS}" V=1 all-libiberty
  COMMAND
    make -C libiberty install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "CC=${CC}"
      --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}

  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py"
)

# This matches the conventions of the FindLibIberty.cmake module.
# Other than that, there is no source of truth for this.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libiberty-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libiberty-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libiberty-config.cmake" DESTINATION lib/cmake/libiberty)
