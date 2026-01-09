if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-mpfr-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: https://www.mpfr.org/mpfr-current/mpfr-4.2.2.tar.xz
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/mpfr-4.2.2.tar.xz"
      URL_HASH "SHA512=eb9e7f51b5385fb349cc4fba3a45ffdf0dd53be6dfc74932dc01258158a10514667960c530c47dd9dfc5aa18be2bd94859d80499844c5713710581e6ac6259a9"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-mpfr
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
        therock-gmp
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
    therock_cmake_subproject_provide_package(therock-mpfr mpfr lib/rocm_sysdeps/lib/cmake/mpfr)
    therock_cmake_subproject_activate(therock-mpfr)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libmpfr.so
    )

    add_test(
      NAME mpfr_header_test_1
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/mpfr.h"
    )

    add_test(
      NAME mpfr_header_test_2
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/mpf2mpfr.h"
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(MPFR_BUILD VERSION "4.2.2")

# MPFR depends on GMP, so we need to pass the GMP prefix to MPFR's configure.
# Extract that information now.

# Create a temporary variable to hold the list we are filtering
# We use CMAKE_PREFIX_PATH as it contains the gmp stage prefix.
set(temp_gmp_paths ${CMAKE_PREFIX_PATH})

# Filter the temporary list to include only the item that contains the gmp
# stage prefix we are looking for.
list(FILTER temp_gmp_paths INCLUDE REGEX "/gmp/build/stage")

# Store our path or complain.
if(temp_gmp_paths)
    list(GET temp_gmp_paths 0 GMP_PREFIX)
    message(STATUS "Extracted GMP_PREFIX: ${GMP_PREFIX}")
else()
    message(SEND_ERROR "Did not find a path containing '/gmp/build/stage' in CMAKE_LIBRARY_PATH.")
endif()

# Do builds from a copy of the source directory, in case we need to
# patch the sources.
set(SOURCE_COPY_DIR "${CMAKE_CURRENT_BINARY_DIR}/s")

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${SOURCE_COPY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${SOURCE_COPY_DIR}"
  #
  # OPTIONAL: Patching of the sources should happen here.
  #
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
      # We don't insert rpath here because there are multiple layers of
      # escaping going on (cmake, configure, shell, libtool), and making
      # our $ORIGIN string survive through all that is complex and
      # potentially fragile. We use a patchelf step instead.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
      --
    "${SOURCE_COPY_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --with-gmp="${GMP_PREFIX}"
  COMMAND
    make V=1
  COMMAND
    make install
  COMMAND
    # Patch in the RUNPATH.
    ${PATCHELF} --set-rpath '\$\$ORIGIN:\$\$ORIGIN/rocm_sysdeps/lib' ${CMAKE_INSTALL_PREFIX}/lib/libmpfr.so
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/mpfr-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/mpfr-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/mpfr-config.cmake" DESTINATION lib/cmake/mpfr)
