# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-util-linux-sources
      SOURCE_DIR "${_source_dir}"
      # Originally downloaded from https://github.com/util-linux/util-linux/archive/refs/tags/v2.42.1.tar.gz
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/util-linux-2.42.1.tar.gz"
      URL_HASH "SHA256=138a3bd9049afa5eeeea73f1e9c0ed7bddf3e55ba4ec29924965a932ec738f01"
      PATCH_COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_tools/patch_third_party_source.py ${CMAKE_SOURCE_DIR}/patches/third-party/util-linux

      TOUCH "${_download_stamp}"
      # We need DOWNLOAD_EXTRACT_TIMESTAMP set to TRUE to preserve the timestamp
      # of the extracted sources, otherwise the build system will notice
      # changes and will try to re-run autotools, causing issues.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-util-linux
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
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-util-linux util-linux lib/rocm_sysdeps/lib/cmake/util-linux)
    therock_cmake_subproject_activate(therock-util-linux)

    # Validate the libraries produced by util-linux (libmount + libblkid).
    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libmount.so libblkid.so
    )

    # Test key headers
    set(UTIL_LINUX_HEADERS "libmount/libmount.h" "blkid/blkid.h")
    foreach(util_linux_header ${UTIL_LINUX_HEADERS})
      string(REPLACE "/" "_" header_test_name "${util_linux_header}")
      add_test(
        NAME util_linux_header_test:${header_test_name}
        COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/${util_linux_header}"
      )
    endforeach()

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(UTIL_LINUX_BUILD VERSION "2.42.1")

if(NOT PATCHELF OR NOT MESON_BUILD)
  message(FATAL_ERROR "Missing PATCHELF or MESON_BUILD from super-project")
endif()

include("${THEROCK_SOURCE_DIR}/cmake/therock_meson_env.cmake")
therock_get_meson_compiler_env(_meson_cc _meson_cxx)

add_custom_target(
  meson_build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"

  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${SOURCE_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}"
  COMMAND
    # Meson refuses to build if the source dir is a subdir of the build dir.
    # Run meson from the source directory to avoid this issue.
    "${CMAKE_COMMAND}" -E chdir "${SOURCE_DIR}"
    "${CMAKE_COMMAND}" -E env
      # Symbol versioning is applied by patch_source.sh, which overwrites
      # libmount.sym and libblkid.sym with version.lds contents. Meson then
      # picks them up via its existing link_args: -Wl,--version-script=...
      # RPATH is set post-install via patch_install.py.
      "CC=${_meson_cc}"
      "CXX=${_meson_cxx}"
      --
      "${MESON_BUILD}" setup "${CMAKE_CURRENT_BINARY_DIR}"
        --reconfigure
        # We generate relocatable, arch neutral directory layouts and use
        # DESTDIR to install into the lib/rocm_sysdeps tree.
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        # openpty() is in libutil on older systems
        -Dc_link_args=-lutil
        # Turn everything off (there's a LOT of libraries and tools) and
        # then enable libmount and its dependency libblkid
        -Dauto_features=disabled
        -Dbuild-libmount=enabled
        -Dbuild-libblkid=enabled
  COMMAND
    "${MESON_BUILD}" compile --verbose
  COMMAND
    "${CMAKE_COMMAND}" -E env "DESTDIR=${CMAKE_INSTALL_PREFIX}" --
      "${MESON_BUILD}" install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/util-linux-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/util-linux-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/util-linux-config.cmake" DESTINATION lib/cmake/util-linux)
