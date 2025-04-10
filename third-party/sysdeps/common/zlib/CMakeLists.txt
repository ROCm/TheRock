if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-zlib-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://www.zlib.net/zlib-1.3.1.tar.gz"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/zlib-1.3.1.tar.gz"
      URL_HASH "SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-zlib
      EXTERNAL_SOURCE_DIR .
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE

      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DPATCHELF=${PATCHELF}"
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
    therock_cmake_subproject_provide_package(therock-zlib ZLIB lib/rocm_sysdeps/lib/cmake/ZLIB)
    therock_cmake_subproject_activate(therock-zlib)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libz.so
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(ZLIB_BUILD)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "LDFLAGS=-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds" --
    "${SOURCE_DIR}/configure" --prefix "${CMAKE_INSTALL_PREFIX}"
  COMMAND
    make -j "${PAR_JOBS}"
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/zlib-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/zlib-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/zlib-config.cmake" DESTINATION lib/cmake/ZLIB)
