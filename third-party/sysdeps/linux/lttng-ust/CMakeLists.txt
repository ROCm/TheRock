# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-lttng-ust-sources
      SOURCE_DIR "${_source_dir}"
      # TODO(sysdeps-mirror): mirror to rocm-third-party-deps S3 before merge.
      # Originally from: https://lttng.org/files/lttng-ust/lttng-ust-2.13.7.tar.bz2
      URL "https://lttng.org/files/lttng-ust/lttng-ust-2.13.7.tar.bz2"
      URL_HASH "SHA256=5fb4f17c307c8c1b79c68561e89be9562d07e7425bf40e728c4d66755342a5eb"
      TOUCH "${_download_stamp}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-lttng-ust
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
      # lttng-ust's configure resolves userspace-rcu via pkg-config (headers /
      # static inlines only -- there is no DT_NEEDED on liburcu), so urcu must
      # build first and its pkgconfig dir must be visible.
      RUNTIME_DEPS
        therock-userspace-rcu
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
    therock_cmake_subproject_activate(therock-lttng-ust)

    # Validate every default-built library so a rename/SONAME regression in
    # patch_source.sh / patch_install.py is caught by CI, not just at runtime.
    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES liblttng-ust.so liblttng-ust-common.so liblttng-ust-tracepoint.so
                liblttng-ust-ctl.so liblttng-ust-dl.so liblttng-ust-fd.so
                liblttng-ust-fork.so liblttng-ust-libc-wrapper.so
                liblttng-ust-pthread-wrapper.so liblttng-ust-cyg-profile.so
                liblttng-ust-cyg-profile-fast.so
    )

    add_test(
      NAME lttng-ust_header_test
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/lttng/tracepoint.h"
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LTTNG_UST_BUILD VERSION "2.13.7")

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

find_program(ACLOCAL aclocal)
if(NOT ACLOCAL)
  message(FATAL_ERROR "lttng-ust cannot be reliably built without autotools (on Ubuntu: apt install automake autoconf libtool)")
endif()

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

# The userspace-rcu sysdep install (its pkgconfig + include dirs) is exposed via
# CMAKE_PREFIX_PATH. Extract it so we can point lttng-ust's autotools configure
# at the bundled urcu rather than a system copy.
function(fetch_library_prefix input_path_list pattern output_var)
  set(_input_list "${${input_path_list}}")
  list(FILTER _input_list INCLUDE REGEX "${pattern}")
  if(_input_list)
    list(GET _input_list 0 _result_path)
    set(${output_var} "${_result_path}" PARENT_SCOPE)
    message(STATUS "Extracted path for pattern '${pattern}': ${_result_path}")
  else()
    message(SEND_ERROR "Did not find a path containing '${pattern}' in CMAKE_PREFIX_PATH.")
  endif()
endfunction()

fetch_library_prefix(CMAKE_PREFIX_PATH "/userspace-rcu/build/stage" URCU_PREFIX)

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    autoreconf -f -i "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    # URCU_PREFIX already resolves to the userspace-rcu install prefix
    # (.../userspace-rcu/build/stage/lib/rocm_sysdeps) -- the dependency
    # provider backs up two levels from the advertised lib/pkgconfig dir -- so
    # do NOT append lib/rocm_sysdeps again. Mirrors hwloc's use of NUMA_PREFIX.
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=${URCU_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
      "CPPFLAGS=-I${URCU_PREFIX}/include"
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -L${URCU_PREFIX}/lib -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/rocm_sysdeps.map"
      --
    "${CMAKE_CURRENT_BINARY_DIR}/s/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --libdir "${CMAKE_INSTALL_PREFIX}/lib"
      --disable-static
      --disable-man-pages
      --disable-numa
      --disable-examples
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
