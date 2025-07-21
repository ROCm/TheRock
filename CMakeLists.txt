if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory.")
endif()

cmake_minimum_required(VERSION 3.25)  # Also update in alternative_os_setup.md

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(THEROCK)

cmake_policy(SET CMP0135 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CMakeDependentOption)
include(ExternalProject)
include(therock_amdgpu_targets)
include(therock_artifacts)
include(therock_compiler_config)
include(therock_features)
include(therock_subproject)
include(therock_job_pools)
include(therock_testing)

################################################################################
# Testing Setup
# Does normal CMake setup so that BUILD_TESTING functions as a master control
# switch. Then we set THEROCK_BUILD_TESTING if not explicitly set by the user
# based on the advice here:
#   https://cliutils.gitlab.io/modern-cmake/chapters/testing.html
# THEROCK_BUILD_TESTING is used to condition inclusion of testing binaries, etc.
################################################################################

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
# Options
################################################################################
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THEROCK_BACKGROUND_BUILD_JOBS "0" CACHE STRING "Number of jobs to reserve for projects marked for background building (empty=auto or a number)")

set(THEROCK_PACKAGE_VERSION "git" CACHE STRING "Sets the package version string")
# Disable compatibility symlinks created in default ROCM cmake project wide
set(ROCM_SYMLINK_LIBS OFF)

set(THEROCK_ARTIFACT_ARCHIVE_SUFFIX "" CACHE STRING "Suffix to add to artifact archive file stem names")

option(THEROCK_BUNDLE_SYSDEPS "Builds bundled system deps for portable builds into lib/rocm_sysdeps" ON)

# Source settings.
# Source can be consumed from the https://github.com/ROCm/rocm-libraries monorepo,
# while the default is to consume the submodules defined in TheRock's `.gitmodules`.
option(THEROCK_USE_EXTERNAL_ROCM_LIBRARIES "Use the `rocm-libraries` monorepo instead of submodules" OFF)
if(THEROCK_USE_EXTERNAL_ROCM_LIBRARIES AND NOT THEROCK_ROCM_LIBRARIES_SOURCE_DIR)
  message(FATAL_ERROR "If THEROCK_USE_EXTERNAL_ROCM_LIBRARIES is set, THEROCK_ROCM_LIBRARIES_SOURCE_DIR is required!")
endif()

if(THEROCK_ROCM_LIBRARIES_SOURCE_DIR)
  cmake_path(ABSOLUTE_PATH THEROCK_ROCM_LIBRARIES_SOURCE_DIR NORMALIZE)
endif()

# Overall build settings.
option(THEROCK_VERBOSE "Enables verbose CMake statuses" OFF)

# Initialize the install directory.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${THEROCK_SOURCE_DIR}/install" CACHE PATH "" FORCE)
  message(STATUS "Defaulted CMAKE_INSTALL_PREFIX to ${CMAKE_INSTALL_PREFIX}")
endif()

block(PROPAGATE ROCM_MAJOR_VERSION ROCM_MINOR_VERSION ROCM_PATCH_VERSION)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.json VERSION_JSON_STRING)
  string(JSON VERSION_STRING GET ${VERSION_JSON_STRING} rocm-version)
  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" ROCM_VERSION ${VERSION_STRING})
  if(ROCM_VERSION)
    set(ROCM_MAJOR_VERSION ${CMAKE_MATCH_1})
    set(ROCM_MINOR_VERSION ${CMAKE_MATCH_2})
    set(ROCM_PATCH_VERSION ${CMAKE_MATCH_3})
    message(STATUS "ROCm version: ${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}")
  else()
    message(FATAL_ERROR "Failed to determine ROCm version.")
  endif()
endblock()

# Various things need to be lined up across layers to precisely match the HIP version.
# The only source of truth for this is from the HIP/VERSION file, so we parse it the
# same way that clr does. While this *should* generally match the ROCM version, at
# least from a major/minor perspective, the ROCM version is a "user" version, whereas
# the HIP version is hardcoded into various places in the code.
block(SCOPE_FOR VARIABLES PROPAGATE THEROCK_HIP_MAJOR_VERSION THEROCK_HIP_MINOR_VERSION)
  set(VERSION_PATH "${CMAKE_CURRENT_SOURCE_DIR}/core/HIP/VERSION")
  if(NOT EXISTS "${VERSION_PATH}")
    message(FATAL_ERROR "Could not find HIP VERSION file: ${VERSION_PATH}")
  endif()
  file(STRINGS "${VERSION_PATH}" VERSION_LIST REGEX "^[0-9]+")
  list(GET VERSION_LIST 0 THEROCK_HIP_MAJOR_VERSION)
  list(GET VERSION_LIST 1 THEROCK_HIP_MINOR_VERSION)
  list(GET VERSION_LIST 2 THEROCK_HIP_PATCH_VERSION)
  message(STATUS "HIP version: ${THEROCK_HIP_MAJOR_VERSION}.${THEROCK_HIP_MINOR_VERSION}.${THEROCK_HIP_PATCH_VERSION}")
endblock()

################################################################################
# Debug Options
################################################################################

option(THEROCK_SPLIT_DEBUG_INFO "Enables splitting of debug info into dbg artifacts (and strips primary packages)" OFF)
option(THEROCK_MINIMAL_DEBUG_INFO "Enables compiler-specific flags for minimal debug symbols suitable for shipping in packages" OFF)
option(THEROCK_QUIET_INSTALL "Enable quiet install logging (install logs only go to the logfile)" ON)
option(THEROCK_USE_SAFE_DEPENDENCY_PROVIDER "Enable the safe dependency provider, performing strict package checks" ON)

################################################################################
# Feature selection
# Each feature added via therock_add_feature produces a boolean cache variable
# like `THEROCK_ENABLE_${feature_name}` that takes its default value from the
# `THEROCK_ENABLE_${group_name}` if a GROUP is present.
################################################################################
option(THEROCK_ENABLE_ALL "Enables building of all feature groups" ON)
option(THEROCK_ENABLE_CORE "Enable building of core libraries" "${THEROCK_ENABLE_ALL}")
option(THEROCK_ENABLE_COMM_LIBS "Enable building of comm libraries" "${THEROCK_ENABLE_ALL}")
option(THEROCK_ENABLE_MATH_LIBS "Enable building of math libraries" "${THEROCK_ENABLE_ALL}")
option(THEROCK_ENABLE_ML_LIBS "Enable building of ML libraries" "${THEROCK_ENABLE_ALL}")
option(THEROCK_ENABLE_PROFILER "Enable building the profiler libraries" "${THEROCK_ENABLE_ALL}")
option(THEROCK_ENABLE_HOST_MATH "Build all bundled host math libraries by default" OFF)
option(THEROCK_RESET_FEATURES "One-shot flag which forces all feature flags to their default state for this configuration run" OFF)

# Host Math Features.
therock_add_feature(HOST_BLAS
  GROUP HOST_MATH
  DESCRIPTION "Bundled host BLAS library"
)
if(NOT WIN32)
  # Currently unsupported on Windows.
  therock_add_feature(HOST_SUITE_SPARSE
    GROUP HOST_MATH
    DESCRIPTION "Bundled SuiteSparse library"
    REQUIRES HOST_BLAS
  )
endif()

# Base Features.
therock_add_feature(COMPILER
  GROUP ALL
  DESCRIPTION "Enables the AMDGPU+host compiler toolchain"
)
therock_add_feature(HIPIFY
  GROUP ALL
  DESCRIPTION "Enables the hipify tool"
  REQUIRES COMPILER
)

# Core Features.
# The HIP runtime for ROCm uses different implementations per platform today:
#   * HIP on Linux uses the HSA runtime in ROCR-Runtime, used by CLR.
#   * HIP on Windows uses the PAL runtime included directly as part of CLR.
# https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/component-support.html
if(NOT WIN32)
  # Other platforms (Linux) _do_ have the "core runtime".
  therock_add_feature(CORE_RUNTIME
    GROUP CORE
    DESCRIPTION "Enables the core device runtime and tools"
  )
  set(_hip_runtime_platform_requirements "CORE_RUNTIME")
else()
  # No "core runtime" (ROCR-Runtime + rocminfo) on Windows.
  # TODO(#36): Could provide an alternate or no-op "core runtime" instead?
  set(_hip_runtime_platform_requirements "")
endif()

therock_add_feature(HIP_RUNTIME
  GROUP CORE
  DESCRIPTION "Enables the HIP runtime"
  REQUIRES COMPILER ${_hip_runtime_platform_requirements}
)

# Profiler Features.
if(NOT WIN32)
  therock_add_feature(ROCPROF_TRACE_DECODER_BINARY
    GROUP PROFILER
    DESCRIPTION "Enables the closed-source rocprof trace decoder binary"
  )
  # Other platforms (Linux) _do_ have the profiler SDK.
  therock_add_feature(ROCPROFV3
    GROUP PROFILER
    DESCRIPTION "Enables the rocprofilerv3 project (and ancillary deps)"
    REQUIRES HIP_RUNTIME
  )
  set(_optional_profiler_requirements "ROCPROFV3")
else()
  # No profiler SDK on Windows.
  set(_optional_profiler_requirements "")
endif()

# Comm-libs Features.
if(NOT WIN32)
  # RCCL is Linux-only.
  therock_add_feature(RCCL
    GROUP COMM_LIBS
    DESCRIPTION "Enables rccl"
    REQUIRES COMPILER HIP_RUNTIME HIPIFY
  )
endif()

# Math-libs Features.
therock_add_feature(PRIM
  GROUP MATH_LIBS
  DESCRIPTION "Enables prim libraries (rocprim, hipCUB, rocThrust)"
  REQUIRES COMPILER HIP_RUNTIME
)
therock_add_feature(BLAS
  GROUP MATH_LIBS
  DESCRIPTION "Enables blas libraries (hipblaslt, rocblas, hipblas)"
  # NOTE: HOST_BLAS is only needed for some tests and PROFILER_SDK is
  # currently only required for rocblas. We may want to re-evaluate the deps
  # if they continue to be sparse like this as it increases minimum dep length.
  REQUIRES COMPILER HIP_RUNTIME HOST_BLAS ${_optional_profiler_requirements}
)
therock_add_feature(RAND
  GROUP MATH_LIBS
  DESCRIPTION "Enables rand libraries (hiprand, rocrand)"
  REQUIRES COMPILER HIP_RUNTIME
)
therock_add_feature(FFT
  GROUP MATH_LIBS
  DESCRIPTION "Enables fft libraries"
  REQUIRES COMPILER HIP_RUNTIME RAND
)
therock_add_feature(SPARSE
  GROUP MATH_LIBS
  DESCRIPTION "Enables sparse libraries (hipsparse, rocsparse)"
  REQUIRES COMPILER HIP_RUNTIME BLAS PRIM ${_optional_profiler_requirements}
)
if(NOT WIN32)
  set(_solver_platform_requirements "HOST_SUITE_SPARSE")
else()
  set(_solver_platform_requirements "")
endif()
therock_add_feature(SOLVER
  GROUP MATH_LIBS
  DESCRIPTION "Enables solver libraries (hipsolver, rocsolver)"
  REQUIRES COMPILER HIP_RUNTIME BLAS PRIM SPARSE ${_solver_platform_requirements}
)

if(NOT WIN32)
  therock_add_feature(COMPOSABLE_KERNEL
    GROUP MATH_LIBS
    DESCRIPTION "Enables build of composable kernel, including minimal kernel library"
    REQUIRES COMPILER HIP_RUNTIME RAND
  )
  cmake_dependent_option(THEROCK_MIOPEN_USE_COMPOSABLE_KERNEL "Enables composable kernel in MIOpen" ON ${THEROCK_ENABLE_COMPOSABLE_KERNEL} OFF)
endif()

# ML-Libs Features.
therock_add_feature(MIOPEN
  GROUP ML_LIBS
  DESCRIPTION "Enables the MIOpen project (with minimal deps defaults to ON)"
  REQUIRES COMPILER HIP_RUNTIME BLAS RAND ${_optional_profiler_requirements}
)

# Finalize all feature flags.
therock_finalize_features()
therock_report_features()

################################################################################
# GPU target selection
#
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
################################################################################

set(THEROCK_AMDGPU_FAMILIES "" CACHE STRING "AMDGPU target families to build for")
set(THEROCK_AMDGPU_TARGETS "" CACHE STRING "AMDGPU targets to build for")
set(THEROCK_AMDGPU_DIST_BUNDLE_NAME "" CACHE STRING "Distribution bundle name for AMDGPU packages")

therock_validate_amdgpu_targets()

################################################################################
# Global setup
################################################################################

# Some sub-projects need Python. Make sure it is found consistently.
find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)

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
# Sysdep bundling
# Each available bundled sysdep is made available with a global variable like
# `THEROCK_BUNDLED_{name}`. This can be included in subproject RUNTIME_DEPS.
# If bundling is enabled, then this will be a target, and otherwise empty.
# Additional platform specific settings may be needed to configure RPATH or
# SxS DLL loading.
# If updating supported libraries here, please update:
#   docs/development/dependencies.md
################################################################################

set(THEROCK_BUNDLED_BZIP2)
set(THEROCK_BUNDLED_ELFUTILS)
set(THEROCK_BUNDLED_LIBDRM)
set(THEROCK_BUNDLED_NUMACTL)
set(THEROCK_BUNDLED_SQLITE3)
set(THEROCK_BUNDLED_ZLIB)
set(THEROCK_BUNDLED_ZSTD)

if(THEROCK_BUNDLE_SYSDEPS)
  message(STATUS "Building with bundled system dependencies enabled")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(PATCHELF patchelf)
    if(NOT PATCHELF)
      message(FATAL_ERROR "Building with THEROCK_BUNDLE_SYSDEPS=ON on Linux requires `patchelf`")
    endif()
    find_program(MESON_BUILD meson)
    if(NOT MESON_BUILD)
      message(FATAL_ERROR "Building with THEROCK_BUNDLE_SYSDEPS=ON on Linux requires (easiest: `pip install meson`)")
    endif()
    set(THEROCK_BUNDLED_BZIP2 therock-bzip2)
    set(THEROCK_BUNDLED_ELFUTILS therock-elfutils)
    set(THEROCK_BUNDLED_LIBDRM therock-libdrm)
    set(THEROCK_BUNDLED_NUMACTL therock-numactl)
    set(THEROCK_BUNDLED_SQLITE3 therock-sqlite3)
    set(THEROCK_BUNDLED_ZLIB therock-zlib)
    set(THEROCK_BUNDLED_ZSTD therock-zstd)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(THEROCK_BUNDLED_BZIP2 therock-bzip2)
    set(THEROCK_BUNDLED_SQLITE3 therock-sqlite3)
    set(THEROCK_BUNDLED_ZLIB therock-zlib)
    set(THEROCK_BUNDLED_ZSTD therock-zstd)
  else()
    message(FATAL_ERROR "Bundled system deps not supported on this platform (THEROCK_BUNDLE_SYSDEPS=ON)")
  endif()
endif()

################################################################################
# Priority build targets.
# By declaring these first as part of ALL, we influence build order a little
# bit, helping to ensure that the critical path projects start as soon as
# possible.
################################################################################

add_custom_target(therock-priority-build ALL)
if(THEROCK_ENABLE_COMPILER)
  add_dependencies(therock-priority-build amd-llvm)
endif()

################################################################################
# External project setup
################################################################################

add_subdirectory(build_tools)

# Add subdirectories in dependency DAG order (which happens to be semi-alpha:
# don't be fooled).
add_subdirectory(third-party)
add_subdirectory(base)
add_subdirectory(compiler)
add_subdirectory(core)
# Note that rocprofiler-register is in base and is what higher level clients
# depend on. The profiler itself is independent.
add_subdirectory(profiler)
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
therock_create_dist()
