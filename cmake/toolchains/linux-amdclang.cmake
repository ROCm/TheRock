# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
set(CMAKE_PREFIX_PATH "${ROCM_PATH}" CACHE PATH "Search path for ROCm packages")

set(ROCM_LLVM_PATH "${ROCM_PATH}/lib/llvm")
set(CMAKE_C_COMPILER "${ROCM_LLVM_PATH}/bin/amdclang" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${ROCM_LLVM_PATH}/bin/amdclang++" CACHE FILEPATH "C++/HIP compiler")
set(CMAKE_Fortran_COMPILER "${ROCM_LLVM_PATH}/bin/flang" CACHE FILEPATH "Fortran compiler")

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable position independent code")
