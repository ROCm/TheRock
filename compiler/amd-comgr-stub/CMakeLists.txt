# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25)

project(amd-comgr-stub
  LANGUAGES C CXX ASM
  VERSION 1.0)

include(CMakePackageConfigHelpers)

set(AMD_COMGR_STAGE_PATH "" CACHE PATH "Path to the dist directory containing the amd-comgr project")
set(IMPLIB_SCRIPT "${THEROCK_SOURCE_DIR}/build_tools/third_party/implib/implib-gen.py" CACHE PATH "Path to implib-gen.py")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter)

if(WIN32)
  message(FATAL_ERROR "amd-comgr-stub not supported/required on Windows")
endif()

message(STATUS "COMGR PATH: ${AMD_COMGR_STAGE_PATH}")
if(NOT EXISTS "${AMD_COMGR_STAGE_PATH}")
  message(FATAL_ERROR "Must set -DAMD_COMGR_STAGE_PATH to a valid install location of amd-comgr (got ${AMD_COMGR_STAGE_PATH})")
endif()
message(STATUS "IMPLIB-GEN: ${IMPLIB_SCRIPT}")
if(NOT EXISTS "${IMPLIB_SCRIPT}")
  message(FATAL_ERROR "Could not find implib-gen.py: ${IMPLIB_SCRIPT}")
endif()

# Generate stub files.
set(TRAMPOLINE_DIR "${CMAKE_CURRENT_BINARY_DIR}/trampoline")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${IMPLIB_SCRIPT}"
    --outdir "${TRAMPOLINE_DIR}"
    --dlopen-callback amd_comgr_stub_dlopen
    --thread-safe
    --lazy-load
    "${AMD_COMGR_STAGE_PATH}/lib/libamd_comgr.so"
  COMMAND_ERROR_IS_FATAL ANY
)

add_library(
  amd_comgr
  STATIC
  "${TRAMPOLINE_DIR}/libamd_comgr.so.tramp.S"
  "${TRAMPOLINE_DIR}/libamd_comgr.so.init.c"
)
set_target_properties(amd_comgr PROPERTIES OUTPUT_NAME "amd_comgr_stub")
target_include_directories(amd_comgr PUBLIC
  "$<BUILD_INTERFACE:${AMD_COMGR_STAGE_PATH}/include>"
  "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(amd_comgr PRIVATE
  amd_comgr_loader
  ${CMAKE_DL_LIBS}
  Threads::Threads
)

add_library(
  amd_comgr_loader
  SHARED
  src/loader.cc
)
target_link_libraries(amd_comgr_loader PRIVATE
  ${CMAKE_DL_LIBS}
)
set_target_properties(amd_comgr_loader PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  INSTALL_RPATH "$ORIGIN"
)

add_executable(
  test_amd_comgr_stub
  src/stub_test.c
)
set_target_properties(test_amd_comgr_stub PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
target_link_libraries(test_amd_comgr_stub PRIVATE amd_comgr)

# Install and exports.
install(
  TARGETS amd_comgr amd_comgr_loader test_amd_comgr_stub
  EXPORT amd_comgr_targets
)
install(
  EXPORT amd_comgr_targets
  FILE amd_comgr-targets.cmake
  DESTINATION "lib/cmake/amd_comgr_stub"
)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/amd_comgr-config.cmake"
  INSTALL_DESTINATION "lib/cmake/amd_comgr_stub"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/amd_comgr-config-version.cmake"
  # We advertise a very high version number so that we always satisfy version
  # checks, given that this is a shared library that is a drop-in for the
  # in-tree comgr.
  VERSION "999.99"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/amd_comgr-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/amd_comgr-config-version.cmake"
  DESTINATION lib/cmake/amd_comgr_stub
)
