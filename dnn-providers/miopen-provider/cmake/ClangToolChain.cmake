# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Platform-specific compiler configuration

if(UNIX)
    set(DEFAULT_ROCM_PATH "/opt/rocm")
    set(DEFAULT_ROCM_LLVM_ROOT "/llvm")
    set(DEFAULT_ROCM_COMPILER_EXTENSION "")
elseif(WIN32)
    set(DEFAULT_ROCM_PATH "C:/dist/therock")
    set(DEFAULT_ROCM_LLVM_ROOT "/lib/llvm")
    set(DEFAULT_ROCM_COMPILER_EXTENSION ".exe")
endif()

if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH ${DEFAULT_ROCM_PATH} CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES "ROCM_PATH")

# Unix/Linux: Use ROCm LLVM Clang
set(ROCM_LLVM_BIN_DIR ${ROCM_PATH}${DEFAULT_ROCM_LLVM_ROOT}/bin)
set(ROCM_LLVM_LIB_DIR ${ROCM_PATH}${DEFAULT_ROCM_LLVM_ROOT}/lib)
set(CMAKE_RC_COMPILER rc.exe)

if(EXISTS ${ROCM_LLVM_BIN_DIR})
    # Set the C and C++ compilers to clang and clang++ with a specific directory hint
    set(CMAKE_C_COMPILER ${ROCM_LLVM_BIN_DIR}/clang${DEFAULT_ROCM_COMPILER_EXTENSION})
    set(CMAKE_CXX_COMPILER ${ROCM_LLVM_BIN_DIR}/clang++${DEFAULT_ROCM_COMPILER_EXTENSION})

    message(STATUS "Using ROCm Clang compilers from ${ROCM_LLVM_BIN_DIR}")
else()
    message(
        FATAL_ERROR
            "The directory ${ROCM_LLVM_BIN_DIR} does not exist. Cannot auto select clang compilers."
    )
endif()
