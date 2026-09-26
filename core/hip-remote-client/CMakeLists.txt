################################################################################
# hip-remote-client
#
# Remote HIP runtime client library.
# This library implements the HIP API by forwarding calls to a remote
# HIP worker service running on a Linux system with AMD GPUs.
#
# Supported platforms: Windows, macOS, Linux (without native AMD GPU)
################################################################################

cmake_minimum_required(VERSION 3.25)
project(hip-remote-client VERSION 1.0.0 LANGUAGES C)

# Build on macOS, Windows, or when explicitly requested
if(NOT APPLE AND NOT WIN32 AND NOT HIP_REMOTE_FORCE_BUILD)
  message(STATUS "hip-remote-client: Skipping (not macOS/Windows, set HIP_REMOTE_FORCE_BUILD=ON to force)")
  return()
endif()

message(STATUS "hip-remote-client: Building remote HIP client library")

################################################################################
# Library target
################################################################################

set(HIP_REMOTE_SOURCES
  src/hip_client.c
  src/hip_api_device.c
  src/hip_api_memory.c
  src/hip_api_stream.c
  src/hip_api_module.c
)

set(HIP_REMOTE_HEADERS
  include/hip_remote/hip_remote_protocol.h
  include/hip_remote/hip_remote_client.h
  include/hip_remote/hip_remote_platform.h
)

add_library(amdhip64 SHARED ${HIP_REMOTE_SOURCES})

target_include_directories(amdhip64
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set library properties
set_target_properties(amdhip64 PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Platform-specific settings
if(APPLE)
  set_target_properties(amdhip64 PROPERTIES
    INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_RPATH ON
    MACOSX_RPATH ON
  )
  target_link_libraries(amdhip64 PRIVATE pthread)
elseif(WIN32)
  target_link_libraries(amdhip64 PRIVATE ws2_32)
  # Export all symbols for the DLL
  set_target_properties(amdhip64 PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
else()
  target_link_libraries(amdhip64 PRIVATE pthread)
endif()

################################################################################
# Install
################################################################################

include(GNUInstallDirs)

install(TARGETS amdhip64
  EXPORT hip-remote-client-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/hip_remote
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create hip-config.cmake for find_package support
include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hip-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/hip-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/hip-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/hip-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip
)

install(EXPORT hip-remote-client-targets
  FILE hip-targets.cmake
  NAMESPACE hip::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip
)

################################################################################
# Tests
################################################################################

if(BUILD_TESTING)
  enable_testing()

  # Basic connectivity test (requires TF_WORKER_HOST to be set)
  add_executable(hip_remote_test_basic tests/test_basic.c)
  target_link_libraries(hip_remote_test_basic PRIVATE amdhip64)
  target_include_directories(hip_remote_test_basic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_basic
    COMMAND hip_remote_test_basic
  )

  # Mark as integration test (needs remote worker)
  set_tests_properties(hip_remote_basic PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )
endif()
