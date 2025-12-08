if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-sqlite3-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://www.sqlite.org/2025/sqlite-amalgamation-3490100.zip"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/sqlite-amalgamation-3490100.zip"
      URL_HASH "SHA256=6cebd1d8403fc58c30e93939b246f3e6e58d0765a5cd50546f16c00fd805d2c3"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-sqlite3
      USE_DIST_AMDGPU_TARGETS
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
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-sqlite3 SQLite3 lib/rocm_sysdeps/lib/cmake/SQLite3)
    therock_cmake_subproject_activate(therock-sqlite3)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libsqlite3.so
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(SQLITE3_BUILD)
include(CMakeDependentOption)

set(BUILD_SHARED_LIBS ON)
set(INSTALL_SHLIB_DIR "lib")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(BUILD_SHARED_LIBS OFF)
  set(INSTALL_SHLIB_DIR "bin")
endif()

find_package(Threads REQUIRED)

add_library(
  rocm_sysdeps_sqlite3
  ${SOURCE_DIR}/sqlite3.c
)

if(NOT WIN32)
    add_custom_command(
    TARGET rocm_sysdeps_sqlite3
    POST_BUILD
    COMMAND
        "${CMAKE_COMMAND}" -E create_symlink librocm_sysdeps_sqlite3.so
        ${CMAKE_CURRENT_BINARY_DIR}/libsqlite3.so
    BYPRODUCTS libsqlite3.so
    )
    target_link_options(rocm_sysdeps_sqlite3 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds")
endif()
target_link_libraries(rocm_sysdeps_sqlite3 PRIVATE
  ${CMAKE_DL_LIBS}
  Threads::Threads
)

install(TARGETS rocm_sysdeps_sqlite3)
if(NOT WIN32)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libsqlite3.so TYPE LIB)
endif()
install(FILES "${SOURCE_DIR}/sqlite3.h" TYPE INCLUDE)
install(FILES "${SOURCE_DIR}/sqlite3ext.h" TYPE INCLUDE)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/sqlite3-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/sqlite3-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sqlite3-config.cmake" DESTINATION lib/cmake/SQLite3)
