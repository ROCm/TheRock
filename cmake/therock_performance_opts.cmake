# therock_performance_opts.cmake
# Maximum performance compiler optimization configuration.
# Provides opt-in aggressive optimization flags for runtime performance.

################################################################################
# THEROCK_ENABLE_PERFORMANCE_MODE
################################################################################
#
# When enabled, applies maximum performance compiler optimizations across
# the entire build. This is an opt-in feature for users who want the best
# runtime performance at the cost of longer build times.
#
# Optimizations Applied:
# - -O3: Maximum optimization level
# - -flto: Link-Time Optimization for whole-program analysis
# - -march=native -mtune=native: CPU-specific optimizations
# - -fvectorize -ftree-vectorize: Auto-vectorization
# - -finline-functions: Aggressive inlining
# - -ffast-math: Fast floating-point math (can affect precision)
# - -fomit-frame-pointer: Better register allocation
# - -DNDEBUG: Disable assertions
#
# Usage:
#   cmake -DTHEROCK_ENABLE_PERFORMANCE_MODE=ON ..
#
# Optional Sub-Flags:
#   -DTHEROCK_PERFORMANCE_NO_FAST_MATH=ON  # Disable fast-math
#
# Profile-Guided Optimization (PGO):
# To use PGO for even better performance:
# 1. Build with -DCMAKE_C_FLAGS="-fprofile-generate" -DCMAKE_CXX_FLAGS="-fprofile-generate"
# 2. Run representative workloads to generate profile data
# 3. Rebuild with -DCMAKE_C_FLAGS="-fprofile-use" -DCMAKE_CXX_FLAGS="-fprofile-use"
#
# Expected Performance Gains:
# - Overall: 30-60% improvement on compute-heavy workloads
# - Build time: 50-100% longer due to LTO
# - Binary size: 10-20% larger due to inlining
#
# Note: Performance mode uses -march=native, making the build non-portable
# to different CPU architectures.
################################################################################

option(THEROCK_ENABLE_PERFORMANCE_MODE
  "Enable maximum performance compiler optimizations (O3, LTO, vectorization)" OFF)

# Option to disable fast-math if needed
cmake_dependent_option(THEROCK_PERFORMANCE_NO_FAST_MATH
  "Disable fast-math optimizations in performance mode" OFF
  "THEROCK_ENABLE_PERFORMANCE_MODE" OFF)

################################################################################
# Function: therock_get_performance_flags
################################################################################
# Returns performance optimization flags as strings.
#
# Arguments:
#   OUT_C_FLAGS       - Output variable for C compiler flags
#   OUT_CXX_FLAGS     - Output variable for C++ compiler flags
#   OUT_LINKER_FLAGS  - Output variable for linker flags
#
# Sets output variables in PARENT_SCOPE. If performance mode is disabled,
# all output variables are set to empty strings.
#
function(therock_get_performance_flags OUT_C_FLAGS OUT_CXX_FLAGS OUT_LINKER_FLAGS)
  if(NOT THEROCK_ENABLE_PERFORMANCE_MODE)
    set(${OUT_C_FLAGS} "" PARENT_SCOPE)
    set(${OUT_CXX_FLAGS} "" PARENT_SCOPE)
    set(${OUT_LINKER_FLAGS} "" PARENT_SCOPE)
    return()
  endif()

  # Initialize flag lists
  set(_c_flags)
  set(_cxx_flags)
  set(_linker_flags)

  ############################################################################
  # Base Optimization Level
  ############################################################################
  list(APPEND _c_flags "-O3")
  list(APPEND _cxx_flags "-O3")

  ############################################################################
  # Link-Time Optimization (LTO)
  ############################################################################
  # Enables whole-program optimization across translation units
  # Better supported on Linux/Unix; Windows MSVC LTO handled separately
  if(NOT WIN32)
    list(APPEND _c_flags "-flto=auto" "-fuse-linker-plugin")
    list(APPEND _cxx_flags "-flto=auto" "-fuse-linker-plugin")
    list(APPEND _linker_flags "-flto=auto" "-fuse-linker-plugin")
  endif()

  ############################################################################
  # CPU-Specific Optimizations
  ############################################################################
  # march=native: Use all instructions available on the build machine's CPU
  # mtune=native: Optimize instruction scheduling for build machine's CPU
  # Note: Makes binaries non-portable to different CPU architectures
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    list(APPEND _c_flags "-march=native" "-mtune=native")
    list(APPEND _cxx_flags "-march=native" "-mtune=native")
  endif()

  ############################################################################
  # Auto-Vectorization
  ############################################################################
  # Enables automatic SIMD vectorization for loops
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang supports both flags
    list(APPEND _c_flags "-fvectorize" "-ftree-vectorize")
    list(APPEND _cxx_flags "-fvectorize" "-ftree-vectorize")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC only supports -ftree-vectorize
    list(APPEND _c_flags "-ftree-vectorize")
    list(APPEND _cxx_flags "-ftree-vectorize")
  endif()

  ############################################################################
  # Function Inlining
  ############################################################################
  # Aggressively inline functions for better performance
  list(APPEND _c_flags "-finline-functions")
  list(APPEND _cxx_flags "-finline-functions")

  ############################################################################
  # Fast Math
  ############################################################################
  # Aggressive floating-point optimizations
  # Note: Can affect numerical precision; disable with THEROCK_PERFORMANCE_NO_FAST_MATH
  if(NOT THEROCK_PERFORMANCE_NO_FAST_MATH)
    list(APPEND _c_flags "-ffast-math")
    list(APPEND _cxx_flags "-ffast-math")
  endif()

  ############################################################################
  # Disable Assertions
  ############################################################################
  # Define NDEBUG to disable assertion checks in Release builds
  list(APPEND _c_flags "-DNDEBUG")
  list(APPEND _cxx_flags "-DNDEBUG")

  ############################################################################
  # Omit Frame Pointer
  ############################################################################
  # Free up a register for better code generation
  # Note: Makes debugging harder but improves performance
  list(APPEND _c_flags "-fomit-frame-pointer")
  list(APPEND _cxx_flags "-fomit-frame-pointer")

  ############################################################################
  # Return Flags
  ############################################################################
  # Join lists into space-separated strings
  list(JOIN _c_flags " " _c_flags_str)
  list(JOIN _cxx_flags " " _cxx_flags_str)
  list(JOIN _linker_flags " " _linker_flags_str)

  # Set output variables in parent scope
  set(${OUT_C_FLAGS} "${_c_flags_str}" PARENT_SCOPE)
  set(${OUT_CXX_FLAGS} "${_cxx_flags_str}" PARENT_SCOPE)
  set(${OUT_LINKER_FLAGS} "${_linker_flags_str}" PARENT_SCOPE)
endfunction()

################################################################################
# Global Flag Propagation
################################################################################
# Apply performance flags globally (following pattern from therock_sanitizers.cmake)
# This ensures flags are applied to the super-project and propagate to subprojects.

if(THEROCK_ENABLE_PERFORMANCE_MODE)
  message(STATUS "Performance mode enabled: Applying aggressive optimizations")

  # Get performance flags
  therock_get_performance_flags(_perf_c_flags _perf_cxx_flags _perf_linker_flags)

  # Append to global flags
  string(APPEND CMAKE_C_FLAGS " ${_perf_c_flags}")
  string(APPEND CMAKE_CXX_FLAGS " ${_perf_cxx_flags}")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " ${_perf_linker_flags}")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " ${_perf_linker_flags}")

  # Set for parent scope (so flags propagate to calling CMakeLists.txt)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" PARENT_SCOPE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" PARENT_SCOPE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" PARENT_SCOPE)

  # Report enabled flags
  message(STATUS "  C flags: ${_perf_c_flags}")
  message(STATUS "  CXX flags: ${_perf_cxx_flags}")
  message(STATUS "  Linker flags: ${_perf_linker_flags}")
endif()
