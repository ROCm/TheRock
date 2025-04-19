if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-bzip2-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/bzip2-1.0.8.tar.gz"
      URL_HASH "SHA512=083f5e675d73f3233c7930ebe20425a533feedeaaa9d8cc86831312a6581cefbe6ed0d08d2fa89be81082f2a5abdabca8b3c080bf97218a1bd59dc118a30b9f3"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-bzip2
      EXTERNAL_SOURCE_DIR .
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
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
    therock_cmake_subproject_provide_package(therock-bzip2 BZip2 lib/rocm_sysdeps/lib/cmake/BZip2)
    therock_cmake_subproject_activate(therock-bzip2)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libbz2.so
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(BZIP2_BUILD)

# bzip2's build barely qualifies as such (it doesn't even install the shared
# library). So we just code it in cmake.

set(BUILD_SHARED_LIBS ON)
set(INSTALL_SHLIB_DIR "lib")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(BUILD_SHARED_LIBS OFF)
  set(INSTALL_SHLIB_DIR "bin")
endif()

add_library(
  rocm_sysdeps_bz2
  ${SOURCE_DIR}/blocksort.c
  ${SOURCE_DIR}/huffman.c
  ${SOURCE_DIR}/crctable.c
  ${SOURCE_DIR}/randtable.c
  ${SOURCE_DIR}/compress.c
  ${SOURCE_DIR}/decompress.c
  ${SOURCE_DIR}/bzlib.c
)

if(NOT WIN32)
  add_custom_command(
    TARGET rocm_sysdeps_bz2
    POST_BUILD
    COMMAND
      "${CMAKE_COMMAND}" -E create_symlink librocm_sysdeps_bz2.so
        ${CMAKE_CURRENT_BINARY_DIR}/libbz2.so
    BYPRODUCTS libbz2.so
  )
endif()

target_link_options(rocm_sysdeps_bz2 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds")

install(
  TARGETS rocm_sysdeps_bz2
)

if(NOT WIN32)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libbz2.so TYPE LIB)
endif()

install(FILES "${SOURCE_DIR}/bzlib.h" TYPE INCLUDE)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/bzip2-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/bzip2-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/bzip2-config.cmake" DESTINATION lib/cmake/BZip2)
