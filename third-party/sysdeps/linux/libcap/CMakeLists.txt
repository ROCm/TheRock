if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # When included in TheRock, we download sources and set up the sub-project.
  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
  set(_download_stamp "${_source_dir}/download.stamp")

  therock_subproject_fetch(therock-libcap-sources
    SOURCE_DIR "${_source_dir}"
    URL "https://git.kernel.org/pub/scm/libs/libcap/libcap.git/snapshot/libcap-2.69.tar.gz"
    URL_HASH "SHA256=9cc2fa3ee744d881629cdac1a1b54c302e7684fda3e3622056218c7681642ffb"
    TOUCH "${_download_stamp}"
  )

  therock_cmake_subproject_declare(therock-libcap
    EXTERNAL_SOURCE_DIR .
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

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

# Set variable for reused path to improve code maintainability
set(_libcap_source_dir "${CMAKE_CURRENT_BINARY_DIR}/s")

# Safety check: Prevent accidental deletion of system directories with 'rm -rf'
# Two-layer protection:
#   1. Blacklist: Reject critical system paths (/, /usr, /usr/local)
#   2. Whitelist: Require path contains "build" (typical build directory pattern)
# This allows flexibility for developers (e.g., /home/user/project/build) while
# preventing accidental deletion of system directories or non-build paths.
if(CMAKE_INSTALL_PREFIX STREQUAL "/" OR
   CMAKE_INSTALL_PREFIX STREQUAL "/usr" OR
   CMAKE_INSTALL_PREFIX STREQUAL "/usr/local" OR
   NOT CMAKE_INSTALL_PREFIX MATCHES "/build")
  message(FATAL_ERROR
    "libcap: CMAKE_INSTALL_PREFIX appears to be outside a build directory.\n"
    "  CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}\n"
    "This is dangerous with 'rm -rf'. Please ensure it's within a build directory.")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${_libcap_source_dir}"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${_libcap_source_dir}"
  COMMAND
    make -C "${_libcap_source_dir}/libcap"
      -j "${PAR_JOBS}"
      prefix="${CMAKE_INSTALL_PREFIX}"
      lib=lib
      GOLANG=no
      PAM_CAP=no
      RAISE_SETFCAP=no
  COMMAND
    make -C "${_libcap_source_dir}/libcap"
      install
      prefix="${CMAKE_INSTALL_PREFIX}"
      lib=lib
      GOLANG=no
      PAM_CAP=no
      RAISE_SETFCAP=no
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
  VERBATIM
)
