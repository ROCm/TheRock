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
