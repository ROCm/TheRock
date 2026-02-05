if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-hwloc-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://download.open-mpi.org/release/hwloc/v1.11/hwloc-1.11.13.tar.bz2"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/hwloc-1.11.13.tar.bz2"
      URL_HASH "SHA256=a4494b7765f517c0990d1c7f09d98cb87755bb6b841e4e2cbfebca1b14bac9c8"
      TOUCH "${_download_stamp}"
      # We need DOWNLOAD_EXTRACT_TIMESTAMP set to TRUE to preserve the timestamp
      # of the extracted sources, otherwise the build system will notice
      # changes and will try to re-run autotools, causing issues.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-hwloc
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
        therock-libpciaccess
        therock-numactl
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
    therock_cmake_subproject_provide_package(therock-hwloc hwloc lib/rocm_sysdeps/lib/cmake/hwloc)
    therock_cmake_subproject_activate(therock-hwloc)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libhwloc.so
    )

    add_test(
      NAME hwloc_header_test_1
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/hwloc.h"
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(HWLOC_BUILD)

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

# Helper function to extract dependency prefixes from CMAKE_PREFIX_PATH.
# Needed for autotools-based builds that require --with-* configure flags.
# (Autotools doesn't use CMAKE_PREFIX_PATH directly, so we must extract and pass explicitly.)
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

# Fetch dependency prefixes for hwloc's configure script.
fetch_library_prefix(CMAKE_PREFIX_PATH "/numactl/build/stage" NUMA_PREFIX)
fetch_library_prefix(CMAKE_PREFIX_PATH "/libpciaccess/build/stage" PCIACCESS_PREFIX)

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  #
  # OPTIONAL: Patching of the sources should happen here.
  #
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=${PCIACCESS_PREFIX}/lib/pkgconfig:${NUMA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
      # Apply symbol versioning via version.lds. RPATH is set post-install via patch_install.py.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -L${PCIACCESS_PREFIX}/lib -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
      "CPPFLAGS=-I${PCIACCESS_PREFIX}/include -I${NUMA_PREFIX}/include"
      --
    "${SOURCE_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --disable-static
      --enable-libnuma
      # CRITICAL for GPU discovery
      --enable-pci
      # CPU topology
      --enable-cpuid
      # Disable unnecessary features
      --disable-opencl
      --disable-cuda
      --disable-nvml
      --disable-gl
      --disable-cairo
      --disable-xml
      --disable-plugin-dlopen
      --disable-plugin-ltdl
  COMMAND
    make V=1
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/hwloc-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hwloc-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hwloc-config.cmake" DESTINATION lib/cmake/hwloc)
