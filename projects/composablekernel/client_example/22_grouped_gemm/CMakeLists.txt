# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_grouped_gemm_fixed_nk_fp16 grouped_gemm_fixed_nk_fp16.cpp)
    target_link_libraries(client_grouped_gemm_fixed_nk_fp16 PRIVATE composable_kernel::device_gemm_operations)

    add_executable(client_grouped_gemm_fixed_nk_fp8 grouped_gemm_fixed_nk_fp8.cpp)
    target_link_libraries(client_grouped_gemm_fixed_nk_fp8 PRIVATE composable_kernel::device_gemm_operations)

    add_executable(client_grouped_gemm_fixed_nk_i8 grouped_gemm_fixed_nk_i8.cpp)
    target_link_libraries(client_grouped_gemm_fixed_nk_i8 PRIVATE composable_kernel::device_gemm_operations)

    add_executable(client_grouped_gemm_fixed_nk_bf16 grouped_gemm_fixed_nk_bf16.cpp)
    target_link_libraries(client_grouped_gemm_fixed_nk_bf16 PRIVATE composable_kernel::device_gemm_operations)
endif()
