################################################################################
# hip-remote-worker
#
# Remote HIP worker service for Linux systems with AMD GPUs.
# This executable handles HIP API requests from remote clients.
################################################################################

cmake_minimum_required(VERSION 3.25)
project(hip-remote-worker VERSION 1.0.0 LANGUAGES C CXX)

# Only build on Linux (requires real HIP runtime)
if(NOT UNIX OR APPLE)
  message(STATUS "hip-remote-worker: Skipping (not Linux)")
  return()
endif()

message(STATUS "hip-remote-worker: Building remote HIP worker service")

################################################################################
# Find HIP
################################################################################

# Try to find HIP via the standard mechanism
find_package(hip QUIET)

if(NOT hip_FOUND)
  # Fall back to environment variable
  if(DEFINED ENV{HIP_PATH})
    set(HIP_PATH "$ENV{HIP_PATH}")
  elseif(EXISTS "/opt/rocm/hip")
    set(HIP_PATH "/opt/rocm/hip")
  elseif(EXISTS "/opt/rocm")
    set(HIP_PATH "/opt/rocm")
  endif()

  if(HIP_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}")
    find_package(hip QUIET)
  endif()
endif()

if(NOT hip_FOUND)
  message(WARNING "hip-remote-worker: HIP not found, skipping build")
  return()
endif()

message(STATUS "hip-remote-worker: Found HIP at ${hip_DIR}")

################################################################################
# Include paths (for protocol header)
################################################################################

# Use the protocol header from hip-remote-client
set(HIP_REMOTE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hip-remote-client/include")

if(NOT EXISTS "${HIP_REMOTE_INCLUDE_DIR}/hip_remote/hip_remote_protocol.h")
  message(WARNING "hip-remote-worker: Protocol header not found, skipping build")
  return()
endif()

################################################################################
# Executable target
################################################################################

add_executable(hip-worker
  src/hip_worker_main.c
)

target_include_directories(hip-worker PRIVATE
  ${HIP_REMOTE_INCLUDE_DIR}
)

target_link_libraries(hip-worker PRIVATE
  hip::host
  pthread
)

set_target_properties(hip-worker PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
)

################################################################################
# Install
################################################################################

include(GNUInstallDirs)

install(TARGETS hip-worker
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

################################################################################
# Systemd service file (optional)
################################################################################

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/hip-worker.service.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hip-worker.service"
  @ONLY
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/hip-worker.service"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/systemd/system"
  OPTIONAL
)
