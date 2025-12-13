# therock_feature_groups.cmake
# Feature group definitions and build flags for TheRock.
# This module defines top-level feature groups that control which components
# are built. Individual artifact features are auto-generated from
# BUILD_TOPOLOGY.toml and included via therock_topology.cmake.

################################################################################
# Feature Group Options
################################################################################
# Each feature group controls a set of related components.
# Individual features within each group can be enabled/disabled independently,
# but default to the group's setting.

# Master switch to enable/disable all features
option(THEROCK_ENABLE_ALL
  "Enables building of all feature groups" ON)

# Core system libraries (HIP runtime, compiler, etc.)
option(THEROCK_ENABLE_CORE
  "Enable building of core libraries" "${THEROCK_ENABLE_ALL}")

# Communication libraries (RCCL, etc.)
option(THEROCK_ENABLE_COMM_LIBS
  "Enable building of comm libraries" "${THEROCK_ENABLE_ALL}")

# Math libraries (rocBLAS, rocFFT, etc.)
option(THEROCK_ENABLE_MATH_LIBS
  "Enable building of math libraries" "${THEROCK_ENABLE_ALL}")

# Machine learning libraries (MIOpen, etc.)
option(THEROCK_ENABLE_ML_LIBS
  "Enable building of ML libraries" "${THEROCK_ENABLE_ALL}")

# Profiler libraries (rocProfiler, etc.)
option(THEROCK_ENABLE_PROFILER
  "Enable building the profiler libraries" "${THEROCK_ENABLE_ALL}")

# Data center tools (RDC, SMI, etc.)
option(THEROCK_ENABLE_DC_TOOLS
  "Enable building of data center tools" "${THEROCK_ENABLE_ALL}")

# Host math libraries (bundled BLAS/LAPACK implementations)
option(THEROCK_ENABLE_HOST_MATH
  "Build all bundled host math libraries by default" OFF)

# One-shot flag to force all feature flags to default state
option(THEROCK_RESET_FEATURES
  "One-shot flag which forces all feature flags to their default state for this configuration run" OFF)

################################################################################
# Feature Flags
################################################################################
# Project-wide flags for controlling build system behavior.
# These are typically used to enable/disable specific build features or
# workarounds for platform-specific issues.

# Note that delay-loading is incompatible with ASAN until a more robust mechanism
# is implemented. See: https://github.com/ROCm/TheRock/issues/1783
cmake_dependent_option(
  THEROCK_FLAG_COMGR_DELAY_LOAD
  "Enables delay loading and linker namespacing of the amd_comgr library (Linux only)"
  ON "LINUX" OFF)

# Flag: -DTHEROCK_FLAG_INCLUDE_PROFILER=OFF
# Note that the profiler is an integral part of the system and disabling it is not
# fully supported. However, in early bringup, it can be useful to disable building
# it, even though that may cause problems in other libraries.
# On some platforms (WIN32 presently), the profiler is always disabled.
cmake_dependent_option(
  THEROCK_FLAG_INCLUDE_PROFILER
  "Allows the profiler to be manually disabled by setting to OFF"
  ON "NOT WIN32" OFF)

################################################################################
# Sub-Feature Dependent Options
################################################################################
# These options control behavior within specific artifacts and depend on
# multiple features being enabled.

# MIOpen with Composable Kernel support
if(NOT WIN32)
  cmake_dependent_option(THEROCK_MIOPEN_USE_COMPOSABLE_KERNEL
    "Enables composable kernel in MIOpen" ON
    "THEROCK_ENABLE_COMPOSABLE_KERNEL" OFF)
endif()

# rocWMMA with rocBLAS validation
cmake_dependent_option(THEROCK_ROCWMMA_USE_ROCBLAS
  "Enables rocBLAS validation in rocWMMA" ON
  "THEROCK_ENABLE_ROCWMMA;THEROCK_ENABLE_BLAS" OFF)

# rocWMMA benchmarks (requires testing enabled)
cmake_dependent_option(THEROCK_ROCWMMA_ENABLE_BENCHMARKS
  "Enables building rocWMMA benchmarks" OFF
  "THEROCK_ENABLE_ROCWMMA;THEROCK_BUILD_TESTING" OFF)
