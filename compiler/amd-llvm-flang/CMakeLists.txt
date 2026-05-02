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

set(LLVM_INCLUDE_TESTS OFF)
set(LLVM_BUILD_TESTS OFF)
set(MLIR_INCLUDE_TESTS OFF)
set(FLANG_INCLUDE_TESTS OFF)
set(LLVM_ENABLE_PROJECTS "")

# OpenMP modules are produced by the standalone offload runtime layer.
set(LLVM_ENABLE_RUNTIMES "openmp")
set(FLANG_RUNTIME_F128_MATH_LIB "libquadmath")

include(${LLVM_MONOREPO_DIR}/flang/cmake/modules/FlangCommon.cmake)
if(FLANG_INCLUDE_QUADMATH_H)
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
