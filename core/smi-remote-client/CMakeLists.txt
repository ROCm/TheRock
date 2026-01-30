################################################################################
# smi-remote-client
#
# Remote AMD SMI client library and CLI tool.
# This library queries GPU metrics from a remote HIP worker service.
################################################################################

cmake_minimum_required(VERSION 3.25)
project(smi-remote-client VERSION 1.0.0 LANGUAGES C)

message(STATUS "smi-remote-client: Building remote SMI client library")

################################################################################
# Include paths
################################################################################

# Use the protocol header from hip-remote-client
set(HIP_REMOTE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hip-remote-client/include")

if(NOT EXISTS "${HIP_REMOTE_INCLUDE_DIR}/hip_remote/hip_remote_protocol.h")
  message(WARNING "smi-remote-client: Protocol header not found, skipping build")
  return()
endif()

################################################################################
# Library target
################################################################################

set(SMI_REMOTE_SOURCES
  src/smi_client.c
)

set(SMI_REMOTE_HEADERS
  include/smi_remote/smi_remote_client.h
)

add_library(amdsmi-remote SHARED ${SMI_REMOTE_SOURCES})

target_include_directories(amdsmi-remote
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${HIP_REMOTE_INCLUDE_DIR}
)

# Set library properties
set_target_properties(amdsmi-remote PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Platform-specific settings
if(APPLE)
  set_target_properties(amdsmi-remote PROPERTIES
    INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_RPATH ON
    MACOSX_RPATH ON
  )
endif()

# Link against system libraries
target_link_libraries(amdsmi-remote PRIVATE
  pthread
)

################################################################################
# CLI Tool
################################################################################

add_executable(smi-remote
  tools/smi-remote.c
)

target_link_libraries(smi-remote PRIVATE
  amdsmi-remote
)

target_include_directories(smi-remote PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${HIP_REMOTE_INCLUDE_DIR}
)

set_target_properties(smi-remote PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
)

################################################################################
# Install
################################################################################

include(GNUInstallDirs)

install(TARGETS amdsmi-remote smi-remote
  EXPORT smi-remote-client-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/smi_remote
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

################################################################################
# Tests
################################################################################

if(BUILD_TESTING)
  enable_testing()

  add_executable(test_smi_basic tests/test_smi_basic.c)
  target_link_libraries(test_smi_basic PRIVATE amdsmi-remote)
  target_include_directories(test_smi_basic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HIP_REMOTE_INCLUDE_DIR}
  )

  add_test(NAME smi_remote_basic
    COMMAND test_smi_basic
  )

  set_tests_properties(smi_remote_basic PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )
endif()
