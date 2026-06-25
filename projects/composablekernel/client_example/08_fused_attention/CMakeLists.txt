# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_fused_attention fused_attention.cpp)
    target_link_libraries(client_fused_attention PRIVATE composable_kernel::device_other_operations composable_kernel::device_gemm_operations)

    add_executable(client_fused_attention_bias fused_attention_bias.cpp)
    target_link_libraries(client_fused_attention_bias PRIVATE composable_kernel::device_other_operations composable_kernel::device_gemm_operations)
endif()
