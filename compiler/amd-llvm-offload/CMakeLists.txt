# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)
project(amd-llvm-offload C CXX ASM)

# Source directory for the LLVM monorepo
set(LLVM_MONOREPO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../amd-llvm")

#=============================================================================
# Find pre-built LLVM to get version info needed by runtimes
#=============================================================================
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

# PACKAGE_VERSION is needed by GetClangResourceDir.cmake
set(PACKAGE_VERSION ${LLVM_PACKAGE_VERSION})

# Disable tests
set(LLVM_INCLUDE_TESTS OFF)

#=============================================================================
# Runtimes configuration
#=============================================================================

# Core runtimes: offload always
set(_runtimes "offload")
set(_amdgcn_runtimes "openmp")  # device openmp runtime

# Conditionally add flang-rt when FLANG is enabled
if(THEROCK_ENABLE_FLANG)
  list(APPEND _runtimes "flang-rt")
  list(APPEND _amdgcn_runtimes "flang-rt")
  set(CMAKE_Fortran_COMPILER_WORKS YES CACHE BOOL "" FORCE)
  set(FLANG_RUNTIME_F128_MATH_LIB "libquadmath")
  set(LIBOMPTARGET_BUILD_DEVICE_FORTRT ON)
  set(RUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBC_PROVIDER "llvm")
  set(RUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBCXX_PROVIDER "llvm")
else()
  set(LIBOMPTARGET_BUILD_DEVICE_FORTRT OFF)
endif()

set(LLVM_ENABLE_RUNTIMES "${_runtimes}" CACHE STRING "" FORCE)

# Offload settings
set(OPENMP_ENABLE_LIBOMPTARGET ON)
set(LIBOMPTARGET_ENABLE_DEBUG ON)
set(LIBOMPTARGET_NO_SANITIZER_AMDGPU ON)
set(OFFLOAD_EXTERNAL_PROJECT_UNIFIED_ROCR ON)

# There is an issue with finding the zstd config built by TheRock when zstd
# is searched for in the llvm config. LLVM has a FindZSTD.cmake that is
# found in module mode, which ultimately fails to locate the library.
set(RUNTIMES_CMAKE_ARGS "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON")

# Device targets
set(LLVM_RUNTIME_TARGETS "default;amdgcn-amd-amdhsa")
set(RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON)
set(RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES "${_amdgcn_runtimes}")

# Setting "LIBOMP_COPY_EXPORTS" to `OFF` "aids parallel builds to not interfere
# with each other" as libomp and generated headers are copied into the original
# source otherwise. Defaults to `ON`.
set(LIBOMP_COPY_EXPORTS OFF)

add_subdirectory("${LLVM_MONOREPO_DIR}/runtimes" "runtimes")
