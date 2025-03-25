if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-zstd-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/zstd-1.5.7.tar.gz"
      URL_HASH "SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-zstd
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
    therock_cmake_subproject_provide_package(therock-zstd zstd lib/rocm_sysdeps/lib/cmake/zstd)
    therock_cmake_subproject_activate(therock-zstd)

    therock_test_validate_shared_lib(
        PATH build/dist/lib/rocm_sysdeps/lib
        LIB_NAMES libzstd.so
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(ZSTD_BUILD)
include(ProcessorCount)
ProcessorCount(PAR_JOBS)

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF or MESON_BUILD from super-project")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    make -j "${PAR_JOBS}" install
        V=1
        PREFIX=${CMAKE_INSTALL_PREFIX}
        # NOTE: This is sloppy: this should be LDFLAGS_DYNLIB, but there is a
        # bug in the Makefiles where it includes those options twice, causing
        # errors. Since CPPFLAGS are always also added to the link command, this
        # works, at the expense of adding a do-nothing flag to the compile
        # commands.
        "CPPFLAGS_DYNLIB=-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
  COMMAND
  "${CMAKE_COMMAND}" -E env
    "PATCHELF=${PATCHELF}"
    "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
    "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
  "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/zstd-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/zstd-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/zstd-config.cmake" DESTINATION lib/cmake/zstd)
