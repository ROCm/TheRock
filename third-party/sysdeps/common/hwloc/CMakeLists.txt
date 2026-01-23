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
project(HWLOC_BUILD VERSION "1.11.13")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

# HWLOC depends on NUMA and libpciaccess, so we need to pass their prefixes to HWLOC's configure.
# Extract that information now.

# Create temporary variables to hold the lists we are filtering
# We use CMAKE_PREFIX_PATH as it contains the stage prefixes.
set(temp_numa_paths ${CMAKE_PREFIX_PATH})
set(temp_pciaccess_paths ${CMAKE_PREFIX_PATH})

# Filter the temporary lists to include only the items that contain the stage
# prefixes we are looking for.
list(FILTER temp_numa_paths INCLUDE REGEX "/numactl/build/stage")
list(FILTER temp_pciaccess_paths INCLUDE REGEX "/libpciaccess/build/stage")

# Store our paths or complain.
if(temp_numa_paths)
    list(GET temp_numa_paths 0 NUMA_PREFIX)
    message(STATUS "Extracted NUMA_PREFIX: ${NUMA_PREFIX}")
else()
    message(SEND_ERROR "Did not find a path containing '/numactl/build/stage' in CMAKE_PREFIX_PATH.")
endif()

if(temp_pciaccess_paths)
    list(GET temp_pciaccess_paths 0 PCIACCESS_PREFIX)
    message(STATUS "Extracted PCIACCESS_PREFIX: ${PCIACCESS_PREFIX}")
else()
    message(SEND_ERROR "Did not find a path containing '/libpciaccess/build/stage' in CMAKE_PREFIX_PATH.")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  #
  # OPTIONAL: Patching of the sources should happen here.
  #
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=${PCIACCESS_PREFIX}/lib/pkgconfig:${NUMA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
      # We don't insert rpath here because there are multiple layers of
      # escaping going on (cmake, configure, shell, libtool), and making
      # our $ORIGIN string survive through all that is complex and
      # potentially fragile. We use a patchelf step instead.
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
    # Patch in the RUNPATH.
    ${PATCHELF} --set-rpath '\$\$ORIGIN:\$\$ORIGIN/rocm_sysdeps/lib' ${CMAKE_INSTALL_PREFIX}/lib/libhwloc.so
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
