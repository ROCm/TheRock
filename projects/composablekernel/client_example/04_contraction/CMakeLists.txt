# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_contraction_scale_fp32 contraction_scale_fp32.cpp)
    target_link_libraries(client_contraction_scale_fp32 PRIVATE composable_kernel::device_other_operations composable_kernel::device_contraction_operations composable_kernel::device_gemm_operations)

    add_executable(client_contraction_bilinear_fp32 contraction_bilinear_fp32.cpp)
    target_link_libraries(client_contraction_bilinear_fp32 PRIVATE composable_kernel::device_other_operations composable_kernel::device_contraction_operations composable_kernel::device_gemm_operations)

    add_executable(client_contraction_scale_fp64 contraction_scale_fp64.cpp)
    target_link_libraries(client_contraction_scale_fp64 PRIVATE composable_kernel::device_other_operations composable_kernel::device_contraction_operations composable_kernel::device_gemm_operations)

    add_executable(client_contraction_bilinear_fp64 contraction_bilinear_fp64.cpp)
    target_link_libraries(client_contraction_bilinear_fp64 PRIVATE composable_kernel::device_other_operations composable_kernel::device_contraction_operations composable_kernel::device_gemm_operations)

    add_executable(contraction_g1m2n3k1_add_xdl_fp16 contraction_g1m2n3k1_add_xdl_fp16.cpp)
    target_link_libraries(contraction_g1m2n3k1_add_xdl_fp16 PRIVATE composable_kernel::device_other_operations composable_kernel::device_contraction_operations composable_kernel::device_gemm_operations)
endif()
