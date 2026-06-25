# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_gemm_add_add_layernorm_naive gemm_add_add_layernorm_naive.cpp)
    target_link_libraries(client_gemm_add_add_layernorm_naive PRIVATE composable_kernel::device_gemm_operations composable_kernel::device_other_operations)

    add_executable(client_gemm_add_relu_add_layernorm_welford gemm_add_relu_add_layernorm_welford.cpp)
    target_link_libraries(client_gemm_add_relu_add_layernorm_welford PRIVATE composable_kernel::device_gemm_operations composable_kernel::device_other_operations)
endif()
