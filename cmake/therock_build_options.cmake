# therock_build_options.cmake
# Build configuration options for TheRock.
# This module centralizes all build-related cache variables and options,
# making it easier to find and modify build configuration settings.

################################################################################
# Global CMake Configuration
################################################################################

# Export compile_commands.json for IDE integration and tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Disable compatibility symlinks created in default ROCM cmake project wide
set(ROCM_SYMLINK_LIBS OFF)

################################################################################
# Build Control Options
################################################################################

# Number of jobs to reserve for projects marked for background building
# Empty or 0 means auto-detect
set(THEROCK_BACKGROUND_BUILD_JOBS "0" CACHE STRING
  "Number of jobs to reserve for projects marked for background building (empty=auto or a number)")

# Package version string (typically "git" for development builds)
set(THEROCK_PACKAGE_VERSION "git" CACHE STRING
  "Sets the package version string")

# Suffix to add to artifact archive file stem names
set(THEROCK_ARTIFACT_ARCHIVE_SUFFIX "" CACHE STRING
  "Suffix to add to artifact archive file stem names")

# Verbose CMake statuses for debugging
option(THEROCK_VERBOSE
  "Enables verbose CMake statuses" OFF)

################################################################################
# Source Location Configuration
################################################################################

# Source settings.
# Use the rocm-libraries superrepo tracked in TheRock's `.gitmodules` or
# allow to specify an alternative source location.
set(THEROCK_ROCM_LIBRARIES_SOURCE_DIR_DEFAULT "${THEROCK_SOURCE_DIR}/rocm-libraries")
set(THEROCK_ROCM_LIBRARIES_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR_DEFAULT}" CACHE STRING
  "Path to rocm-libraries superrepo")
cmake_path(ABSOLUTE_PATH THEROCK_ROCM_LIBRARIES_SOURCE_DIR NORMALIZE)

set(THEROCM_ROCM_SYSTEMS_SOURCE_DIR_DEFAULT "${THEROCK_SOURCE_DIR}/rocm-systems")
set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "${THEROCM_ROCM_SYSTEMS_SOURCE_DIR_DEFAULT}" CACHE STRING
  "Path to rocm-systems superrepo")
cmake_path(ABSOLUTE_PATH THEROCK_ROCM_SYSTEMS_SOURCE_DIR NORMALIZE)

# Allow to specify alternative source locations instead of using
# repositories tracked in TheRock's `.gitmodules`.
therock_enable_external_source("rccl" "${THEROCK_SOURCE_DIR}/comm-libs/rccl" OFF)
therock_enable_external_source("rccl-tests" "${THEROCK_SOURCE_DIR}/comm-libs/rccl-tests" OFF)
therock_enable_external_source("composable-kernel" "${THEROCK_SOURCE_DIR}/ml-libs/composable_kernel" OFF)

################################################################################
# Python Build Configuration
################################################################################

# List of python executables to use for multi-version python builds
# If empty, defaults to Python3_EXECUTABLE for single-version builds
set(THEROCK_DIST_PYTHON_EXECUTABLES "" CACHE STRING
  "Build for multiple python versions in supported projects (default to Python3_EXECUTABLE if empty)")

################################################################################
# Install Configuration
################################################################################

# Initialize the install directory to a default location if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${THEROCK_SOURCE_DIR}/install" CACHE PATH "" FORCE)
  message(STATUS "Defaulted CMAKE_INSTALL_PREFIX to ${CMAKE_INSTALL_PREFIX}")
endif()

################################################################################
# Cross-Cutting Feature Options
################################################################################

# Bundled system dependencies for portable builds
option(THEROCK_BUNDLE_SYSDEPS
  "Builds bundled system deps for portable builds into lib/rocm_sysdeps" ON)

# Message Passing Interface (MPI) support
option(THEROCK_ENABLE_MPI
  "Enables building components with Message Passing Interface (MPI) support" OFF)

################################################################################
# Sanitizer Configuration
################################################################################

# Enable project wide sanitizer build (e.g. 'ASAN', 'TSAN', 'UBSAN')
set(THEROCK_SANITIZER "" CACHE STRING
  "Enable project wide sanitizer build (e.g. 'ASAN')")

################################################################################
# Debug Options
################################################################################

# Split debug info into separate .dbg artifacts and strip primary packages
option(THEROCK_SPLIT_DEBUG_INFO
  "Enables splitting of debug info into dbg artifacts (and strips primary packages)" OFF)

# Minimal debug symbols suitable for shipping in packages
option(THEROCK_MINIMAL_DEBUG_INFO
  "Enables compiler-specific flags for minimal debug symbols suitable for shipping in packages" OFF)

# Debug build of compiler-rt
option(THEROCK_COMPILER_RT_DEBUG
  "Enables compiler-rt debug build" OFF)

# Quiet install logging (install logs only go to the logfile)
option(THEROCK_QUIET_INSTALL
  "Enable quiet install logging (install logs only go to the logfile)" ON)

# Safe dependency provider with strict package checks
option(THEROCK_USE_SAFE_DEPENDENCY_PROVIDER
  "Enable the safe dependency provider, performing strict package checks" ON)
