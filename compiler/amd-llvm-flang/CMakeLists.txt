# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)
project(amd-llvm-flang C CXX ASM)

set(LLVM_MONOREPO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../amd-llvm")
set(LLVM_COMMON_CMAKE_UTILS "${LLVM_MONOREPO_DIR}/cmake")

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${CLANG_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH "${LLVM_COMMON_CMAKE_UTILS}/Modules")

include(HandleLLVMOptions)
include(AddLLVM)
include(TableGen)
include(CMakePushCheckState)
include(AddClang)

set(LLVM_MAIN_SRC_DIR "${LLVM_MONOREPO_DIR}/llvm")
set(LLVM_SOURCE_DIR "${LLVM_MONOREPO_DIR}/llvm")
set(CLANG_SOURCE_DIR "${LLVM_MONOREPO_DIR}/clang")

# Preserve the find_package(LLVM) install tree for the OpenMP-modules
# ExternalProject. This file reassigns LLVM_BINARY_DIR to the local build tree.
set(AMD_LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
set(LLVM_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(LLVM_TOOLS_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX}")

set(CLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
set(CLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
set(CLANG_VERSION_PATCH ${LLVM_VERSION_PATCH})
set(PACKAGE_VERSION ${LLVM_PACKAGE_VERSION})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${LLVM_MONOREPO_DIR}/mlir/include)
include_directories(${LLVM_MONOREPO_DIR}/clang/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/clang/include)

set(LLVM_ENABLE_PROJECTS "")
# Test toggles / F128 come from flang-standalone.cmake (-C).

include(${LLVM_MONOREPO_DIR}/flang/cmake/modules/FlangCommon.cmake)
if(FLANG_INCLUDE_QUADMATH_H)
  # TODO: Standalone Flang should expose this through its CMake configuration.
  # Until then, mirror the normal LLVM build's generated quadmath wrapper.
  set(QUADMATH_WRAPPER_DIR "${CMAKE_BINARY_DIR}/include")
  configure_file(
    "${LLVM_MONOREPO_DIR}/flang/cmake/quadmath_wrapper.h.in"
    "${QUADMATH_WRAPPER_DIR}/quadmath_wrapper.h"
  )
  include_directories(${QUADMATH_WRAPPER_DIR})
endif()

set(MLIR_STANDALONE_BUILD OFF)
set(FLANG_STANDALONE_BUILD OFF)

set(MLIR_MAIN_SRC_DIR "${LLVM_MONOREPO_DIR}/mlir")
set(MLIR_SOURCE_DIR "${LLVM_MONOREPO_DIR}/mlir")
set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}/tools/mlir")
set(MLIR_LINK_MLIR_DYLIB OFF)
# LLVM_ENABLE_PER_TARGET_RUNTIME_DIR is set in flang-standalone-runtimes.cmake.

add_subdirectory(${LLVM_MONOREPO_DIR}/mlir tools/mlir EXCLUDE_FROM_ALL)
add_subdirectory(${LLVM_MONOREPO_DIR}/flang flang)

if(TARGET FortranSupport AND TARGET MLIRBuiltinTypeInterfacesIncGen)
  add_dependencies(FortranSupport MLIRBuiltinTypeInterfacesIncGen)
endif()
if(TARGET FIRSupport AND TARGET MLIRLinalgOpsIncGen)
  add_dependencies(FIRSupport MLIRLinalgOpsIncGen)
endif()
if(TARGET FortranSupport)
  target_include_directories(FortranSupport PRIVATE
    ${CMAKE_BINARY_DIR}/flang/lib/Support)
endif()

##############################################################################
# Flang-RT + OpenMP Fortran modules (iso_c_binding.mod, omp_lib.mod, ...)
#
# libomp itself comes from amd-llvm. Modules must be compiled with this
# Flang (matching ABI / .mod format), so build them here after `flang` exists.
# Policy lives in clang/cmake/caches/flang-standalone-runtimes.cmake.
# Only build-tree paths remain as CMAKE_ARGS here.
##############################################################################
include(ExternalProject)

set(_flang_exe "${CMAKE_BINARY_DIR}/bin/flang${CMAKE_EXECUTABLE_SUFFIX}")
set(_openmp_mod_bin_dir "${CMAKE_BINARY_DIR}/runtimes-openmp-mod")

if(NOT FLANG_STANDALONE_RUNTIMES_CACHE)
  message(FATAL_ERROR
    "FLANG_STANDALONE_RUNTIMES_CACHE is unset. It is defined by "
    "flang-standalone.cmake and names the -C file for the runtimes "
    "ExternalProject. Configure amd-llvm-flang with "
    "-C <llvm>/clang/cmake/caches/flang-standalone.cmake")
endif()

ExternalProject_Add(flang-openmp-modules
  SOURCE_DIR "${LLVM_MONOREPO_DIR}/runtimes"
  BINARY_DIR "${_openmp_mod_bin_dir}"
  DEPENDS flang
  CMAKE_ARGS
    -C${FLANG_STANDALONE_RUNTIMES_CACHE}
    -G "${CMAKE_GENERATOR}"
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_Fortran_COMPILER=${_flang_exe}
    -DLLVM_BINARY_DIR=${AMD_LLVM_BINARY_DIR}
  # Both module sets use EXCLUDE_FROM_ALL install components.
  INSTALL_COMMAND
    "${CMAKE_COMMAND}" --install "${_openmp_mod_bin_dir}" --component flang-rt-mod
  COMMAND
    "${CMAKE_COMMAND}" --install "${_openmp_mod_bin_dir}" --component libomp-mod
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE
  USES_TERMINAL_INSTALL TRUE
)

# Ensure `ninja` / the TheRock build step always builds+installs the modules.
add_custom_target(amd-llvm-flang-openmp-modules ALL
  DEPENDS flang-openmp-modules
)
