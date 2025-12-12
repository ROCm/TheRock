if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # When included in TheRock, we download sources and set up the sub-project.
  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
  set(_download_stamp "${_source_dir}/download.stamp")

  therock_subproject_fetch(therock-libcap-sources
    SOURCE_DIR "${_source_dir}"
    # Originally mirrored from: https://git.kernel.org/pub/scm/libs/libcap/libcap.git/snapshot/libcap-2.69.tar.gz
    URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libcap-2.69.tar.gz"
    URL_HASH "SHA256=9cc2fa3ee744d881629cdac1a1b54c302e7684fda3e3622056218c7681642ffb"
    TOUCH "${_download_stamp}"
  )

  therock_cmake_subproject_declare(therock-libcap
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
      "-DTHEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
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
  therock_cmake_subproject_provide_package(therock-libcap Libcap lib/rocm_sysdeps/lib/cmake/Libcap)
  therock_cmake_subproject_activate(therock-libcap)

  therock_test_validate_shared_lib(
    PATH build/dist/lib/rocm_sysdeps/lib
    LIB_NAMES libcap.so
  )
  return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBCAP_BUILD)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)
if(NOT PAR_JOBS OR PAR_JOBS EQUAL 0)
  set(PAR_JOBS 1)
endif()

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

set(_libcap_make_args
  prefix=${CMAKE_INSTALL_PREFIX}
  lib=lib
  GOLANG=no
  PAM_CAP=no
  RAISE_SETFCAP=no
)

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    # We have to patch the sources so make a fresh copy.
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    # TODO: Add symbol versioning patches
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    # Build libcap using its Makefile
    make -C "${CMAKE_CURRENT_BINARY_DIR}/s" -j "${PAR_JOBS}" ${_libcap_make_args}
  COMMAND
    # Install libcap
    make -C "${CMAKE_CURRENT_BINARY_DIR}/s" install ${_libcap_make_args}
  COMMAND
    # Patch installed libraries for symbol prefixing
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" "${CMAKE_INSTALL_PREFIX}"

  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/libcap.map"
  VERBATIM
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libcap-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libcap-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libcap-config.cmake" DESTINATION lib/cmake/Libcap)
