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
