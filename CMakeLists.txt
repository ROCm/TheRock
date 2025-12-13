################################################################################
# Safety Checks and Initialization
################################################################################

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory.")
endif()

cmake_minimum_required(VERSION 3.25)  # Also update in alternative_os_setup.md

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

################################################################################
# Project Declaration
################################################################################

project(THEROCK)

cmake_policy(SET CMP0135 NEW)

################################################################################
# Module Path and Core Includes
################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CMakeDependentOption)
include(ExternalProject)

# Core infrastructure modules
include(therock_globals)
include(therock_default_targets)
include(therock_compiler_config)
include(therock_sanitizers)
include(therock_job_pools)
include(therock_testing)

# Build system modules
include(therock_amdgpu_targets)
include(therock_artifacts)
include(therock_subproject)
include(therock_external_source)
include(therock_features)

# Configuration modules (new)
include(therock_version_info)
include(therock_python_setup)
include(therock_build_options)
include(therock_feature_groups)
include(therock_bundled_sysdeps)
include(therock_performance_opts)

# Feature detection
include(therock_detect_python_versions)

################################################################################
# Python and Build Topology
################################################################################
# Python is used throughout the build. The topology generation creates
# feature definitions from BUILD_TOPOLOGY.toml.

therock_setup_python_and_topology()

################################################################################
# Version Information
################################################################################
# Parse ROCm version from version.json

therock_parse_version_file(
  FILE "${CMAKE_CURRENT_SOURCE_DIR}/version.json"
  TYPE JSON
  PREFIX ROCM
)

# Various things need to be lined up across layers to precisely match the HIP version.
# The only source of truth for this is from the HIP/VERSION file, so we parse it the
# same way that clr does. While this *should* generally match the ROCM version, at
# least from a major/minor perspective, the ROCM version is a "user" version, whereas
# the HIP version is hardcoded into various places in the code.

therock_parse_version_file(
  FILE "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hip/VERSION"
  TYPE LIST
  PREFIX THEROCK_HIP
)

################################################################################
# Testing Setup
################################################################################
# Does normal CMake setup so that BUILD_TESTING functions as a master control
# switch. Then we set THEROCK_BUILD_TESTING if not explicitly set by the user
# based on the advice here:
#   https://cliutils.gitlab.io/modern-cmake/chapters/testing.html
# THEROCK_BUILD_TESTING is used to condition inclusion of testing binaries, etc.

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND NOT DEFINED THEROCK_BUILD_TESTING)
  set(THEROCK_BUILD_TESTING ${BUILD_TESTING})
endif()
if(THEROCK_BUILD_TESTING)
  message(STATUS "Enabling building tests")
endif()

################################################################################
# Notices
################################################################################
# If there are ongoing issues for specific configurations log them here.
#
#   message(WARNING
#     "*************************************************\n"
#     ".."
#     " The last known good commit on {Windows,Linux} is"
#     " https://github.com/ROCm/TheRock/commit/{HASH}"
#     " See https://github.com/ROCm/TheRock/issues/{ISSUE} for details.\n"
#     "*************************************************")

################################################################################
# Build Options and Configuration
################################################################################
# Build options are defined in cmake/therock_build_options.cmake
# This includes:
#   - Global CMake settings (EXPORT_COMPILE_COMMANDS, INSTALL_LIBDIR)
#   - Build control options (BACKGROUND_BUILD_JOBS, PACKAGE_VERSION, etc.)
#   - Source directory settings (ROCM_LIBRARIES_SOURCE_DIR, ROCM_SYSTEMS_SOURCE_DIR)
#   - Install prefix initialization
#   - External source declarations
#   - Debug and sanitizer options

################################################################################
# Feature Groups and Flags
################################################################################
# Feature groups are defined in cmake/therock_feature_groups.cmake
# This includes:
#   - THEROCK_ENABLE_ALL and group-level options (CORE, COMM_LIBS, etc.)
#   - THEROCK_FLAG_* definitions (COMGR_DELAY_LOAD, INCLUDE_PROFILER)
#   - Sub-feature dependent options
#
# Individual artifact features are auto-generated from BUILD_TOPOLOGY.toml
# To modify artifact features, edit BUILD_TOPOLOGY.toml

# Include generated topology (contains feature definitions)
include("${CMAKE_CURRENT_BINARY_DIR}/cmake/therock_topology.cmake")

################################################################################
# Manual Features (not from topology)
################################################################################

# Profiler Trace Decoder Binary (not an artifact, just a config option)
if(THEROCK_FLAG_INCLUDE_PROFILER)
  therock_add_feature(ROCPROF_TRACE_DECODER_BINARY
    GROUP PROFILER
    DESCRIPTION "Enables the closed-source rocprof trace decoder binary"
  )
endif()

# Optional sub-features that control behavior within artifacts

# Math library sub-components (control what gets built within BLAS artifact)
therock_add_feature(SPARSE
  GROUP MATH_LIBS
  DESCRIPTION "Enables sparse libraries (hipsparse, hipsparselt, rocsparse) within BLAS"
  REQUIRES BLAS PRIM
)

if(NOT WIN32)
  set(_solver_platform_requirements "HOST_SUITE_SPARSE")
else()
  set(_solver_platform_requirements "")
endif()
therock_add_feature(SOLVER
  GROUP MATH_LIBS
  DESCRIPTION "Enables solver libraries (hipsolver, rocsolver) within BLAS"
  REQUIRES BLAS PRIM SPARSE ${_solver_platform_requirements}
)


# Finalize all feature flags.
therock_finalize_features()
therock_report_features()

################################################################################
# GPU Target Selection
################################################################################
# GPU target selection can be done by specifying one of THEROCK_AMDGPU_FAMILIES
# or THEROCK_AMDGPU_TARGETS. Most targets are bundled into families that include
# several related targets.
#
# If exactly one family or target is specified, then that is also taken to be
# the THEROCK_AMDGPU_DIST_BUNDLE_NAME, if omitted (this is the identifier
# embedded into package names). If more than one family or discrete target is
# specified, then the bundle name must be specified manually.
#
# Once cache variable validation is done, THEROCK_AMDGPU_TARGETS will be the
# fully expanded list of targets (as a local variable). For convenience and
# because some parts of the tree use a space separated list,
# THEROCK_AMDGPU_TARGETS_SPACES will also be set.
#
# See therock_amdgpu_targets.cmake for further details.

set(THEROCK_AMDGPU_FAMILIES "" CACHE STRING "AMDGPU target families to build for")
set(THEROCK_AMDGPU_TARGETS "" CACHE STRING "AMDGPU targets to build for")
set(THEROCK_AMDGPU_DIST_BUNDLE_NAME "" CACHE STRING "Distribution bundle name for AMDGPU packages")

therock_validate_amdgpu_targets()

################################################################################
# Global Setup
################################################################################

set(STAGING_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/staging_install")

# On some distributions, this will install to lib64. We would like
# consistency in built packages, so hard-code it.
set(CMAKE_INSTALL_LIBDIR "lib")

if(CMAKE_C_VISIBILITY_PRESET)
  list(APPEND DEFAULT_CMAKE_ARGS ${CMAKE_C_VISIBILITY_PRESET})
endif()
if(CMAKE_CXX_VISIBILITY_PRESET)
  list(APPEND DEFAULT_CMAKE_ARGS ${CMAKE_CXX_VISIBILITY_PRESET})
endif()

################################################################################
# Bundled System Dependencies
################################################################################
# Bundled sysdep configuration is in cmake/therock_bundled_sysdeps.cmake
# This creates THEROCK_BUNDLED_* variables for use in subproject RUNTIME_DEPS.
#
# Each available bundled sysdep is made available with a global variable like
# `THEROCK_BUNDLED_{name}`. This can be included in subproject RUNTIME_DEPS.
# If bundling is enabled, then this will be a target, and otherwise empty.
# Additional platform specific settings may be needed to configure RPATH or
# SxS DLL loading.
#
# If updating supported libraries here, please update:
#   docs/development/dependencies.md

################################################################################
# Priority Build Targets
################################################################################
# By declaring these first as part of ALL, we influence build order a little
# bit, helping to ensure that the critical path projects start as soon as
# possible.

add_custom_target(therock-priority-build ALL)
if(THEROCK_ENABLE_COMPILER)
  add_dependencies(therock-priority-build amd-llvm)
endif()

################################################################################
# External Project Setup
################################################################################
# Add subdirectories in dependency DAG order (which happens to be semi-alpha:
# don't be fooled).

add_subdirectory(third-party)
add_subdirectory(base)
add_subdirectory(compiler)
add_subdirectory(core)
# Note that rocprofiler-register is in base and is what higher level clients
# depend on. The profiler itself is independent.
add_subdirectory(profiler)
# Data center tools (RDC, etc.)
add_subdirectory(dctools)
add_subdirectory(comm-libs)
add_subdirectory(math-libs)
add_subdirectory(ml-libs)

if(THEROCK_BUILD_TESTING)
  add_subdirectory(examples)
endif()

################################################################################
# Testing
################################################################################

if(THEROCK_BUILD_TESTING)
  if(NOT WIN32)
    add_executable(
      dlopen-hip
      tests/dlopen-hip.c
    )
    target_link_libraries(dlopen-hip dl)
  else()
    # TODO: Test that is compatible with Windows (LoadLibraryA instead of dlopen)
    #   then add instructions in the README to run with a command like
    #   `./build/dlopen-hip install/amdhip64.dll`
  endif()
endif()

################################################################################
# Finalization
################################################################################

therock_subproject_merge_compile_commands()
