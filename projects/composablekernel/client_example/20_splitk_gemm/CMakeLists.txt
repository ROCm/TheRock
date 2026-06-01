# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "fp16") OR (NOT DEFINED DTYPES AND GPU_TARGETS MATCHES "gfx94"))
  add_executable(client_splitK_gemm splitK_gemm_fp16_f8.cpp)
  target_link_libraries(client_splitK_gemm PRIVATE composable_kernel::device_gemm_operations)
endif()
