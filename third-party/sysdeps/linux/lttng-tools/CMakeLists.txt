# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# lttng-tools: the LTTng session daemon, consumer daemon, relay daemon, the lttng command
# and liblttng-ctl. The out-of-process collector (rocprofiler-sdk) starts a private session
# daemon and controls it with liblttng-ctl, so the daemon version matches the bundled
# lttng-ust. Kernel tracing is not built (no kmod); man pages, tests and examples are not
# built.

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-lttng-tools-sources
      SOURCE_DIR "${_source_dir}"
      # TODO(sysdeps-mirror): mirror to rocm-third-party-deps S3 before merge.
      # Originally from: https://lttng.org/files/lttng-tools/lttng-tools-2.13.15.tar.bz2
      URL "https://lttng.org/files/lttng-tools/lttng-tools-2.13.15.tar.bz2"
      URL_HASH "SHA256=96ea42351ee112c19dad9fdc7aae93b583d9f1722b2175664a381d2d337703c4"
      TOUCH "${_download_stamp}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-lttng-tools
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
      RUNTIME_DEPS
        therock-userspace-rcu
        therock-lttng-ust
        therock-popt
        therock-libxml2
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      INTERFACE_INCLUDE_DIRS
        lib/rocm_sysdeps/include
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_activate(therock-lttng-tools)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES liblttng-ctl.so
    )

    # The daemons and the command run from the install tree, finding the bundled
    # libraries through their RPATH.
    foreach(_bin lttng lttng-sessiond lttng-relayd)
      add_test(
        NAME lttng-tools_${_bin}_version_test
        COMMAND "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/bin/${_bin}" --version
      )
    endforeach()
    add_test(
      NAME lttng-tools_consumerd_version_test
      COMMAND "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/lib/lttng/libexec/lttng-consumerd" --version
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LTTNG_TOOLS_BUILD VERSION "2.13.15")

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

find_program(ACLOCAL aclocal)
if(NOT ACLOCAL)
  message(FATAL_ERROR "lttng-tools cannot be reliably built without autotools (on Ubuntu: apt install automake autoconf libtool)")
endif()

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

# The bundled dependencies' install prefixes are in CMAKE_PREFIX_PATH. Point configure's
# pkg-config, preprocessor and linker at them rather than at system copies.
set(_pkg_config_path)
set(_cppflags)
set(_ldflags)
foreach(_dep userspace-rcu lttng-ust popt libxml2)
  set(_matches "${CMAKE_PREFIX_PATH}")
  list(FILTER _matches INCLUDE REGEX "/${_dep}/build/stage")
  if(NOT _matches)
    message(FATAL_ERROR "Did not find the ${_dep} sysdep in CMAKE_PREFIX_PATH")
  endif()
  list(GET _matches 0 _prefix)
  message(STATUS "lttng-tools: ${_dep} at ${_prefix}")
  string(APPEND _pkg_config_path "${_prefix}/lib/pkgconfig:")
  string(APPEND _cppflags " -I${_prefix}/include")
  # rpath-link: the linker resolves the dependencies' own dependencies (urcu under
  # lttng-ust-ctl) from these directories when linking the executables.
  string(APPEND _ldflags " -L${_prefix}/lib -Wl,-rpath-link,${_prefix}/lib")
endforeach()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    # cp -a keeps the timestamps, so the parsers and lexers the release ships (filter
    # parser) stay newer than their sources and bison and flex are not needed.
    cp -a "${SOURCE_DIR}/." "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    autoreconf -f -i "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=${_pkg_config_path}$ENV{PKG_CONFIG_PATH}"
      "CPPFLAGS=${_cppflags}"
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS}${_ldflags} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/rocm_sysdeps.map"
      --
    "${CMAKE_CURRENT_BINARY_DIR}/s/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --libdir "${CMAKE_INSTALL_PREFIX}/lib"
      --disable-static
      --disable-man-pages
      --without-kmod
  COMMAND
    make -j "${PAR_JOBS}" V=1
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)
