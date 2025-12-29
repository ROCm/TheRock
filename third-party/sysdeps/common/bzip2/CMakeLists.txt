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
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_PROGRAM_DIRS
        lib/rocm_sysdeps/bin
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

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(BZ2_TYPE "SHARED")
  set(BZ2_IMPORTED_LOCATION "lib/librocm_sysdeps_bz2.so")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(BZ2_TYPE "STATIC")
  set(BZ2_IMPORTED_LOCATION "lib/rocm_sysdeps_bz2.lib")
else()
  message(FATAL_ERROR "Unrecognized system type")
endif()

add_library(
  bz2_objects
  OBJECT
  ${SOURCE_DIR}/blocksort.c
  ${SOURCE_DIR}/huffman.c
  ${SOURCE_DIR}/crctable.c
  ${SOURCE_DIR}/randtable.c
  ${SOURCE_DIR}/compress.c
  ${SOURCE_DIR}/decompress.c
  ${SOURCE_DIR}/bzlib.c
)

# Build as PIC no matter whether shared/static, since we want it consumable
# in shared libs.
set_target_properties(bz2_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(
  rocm_sysdeps_bz2
  "${BZ2_TYPE}"
)
target_link_libraries(rocm_sysdeps_bz2 PRIVATE bz2_objects)

# We also provide linked bzip2 executables that we include in the
# development package. This will hardly ever be needed on Linux but is somewhat
# harder to ensure on Windows, so we just build it.
foreach(bz2_exe_name bzip2 bzcat bunzip2)
  add_executable(
    "${bz2_exe_name}"
    ${SOURCE_DIR}/bzip2.c
  )
  target_link_libraries("${bz2_exe_name}" PRIVATE rocm_sysdeps_bz2)
  set_target_properties("${bz2_exe_name}" PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
  install(TARGETS "${bz2_exe_name}")
endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_custom_command(
    TARGET rocm_sysdeps_bz2
    POST_BUILD
    COMMAND
      "${CMAKE_COMMAND}" -E create_symlink librocm_sysdeps_bz2.so
        ${CMAKE_CURRENT_BINARY_DIR}/libbz2.so
    BYPRODUCTS libbz2.so
  )
  target_link_options(rocm_sysdeps_bz2 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libbz2.so TYPE LIB)
endif()

install(TARGETS rocm_sysdeps_bz2)
install(FILES "${SOURCE_DIR}/bzlib.h" TYPE INCLUDE)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/bzip2-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/bzip2-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/bzip2-config.cmake" DESTINATION lib/cmake/BZip2)
