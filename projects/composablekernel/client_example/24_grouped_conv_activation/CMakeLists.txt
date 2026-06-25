# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
# Fwd scaleadd scaleadd relu
add_executable(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_fp32
               grouped_convnd_fwd_scaleadd_scaleadd_relu/grouped_conv_fwd_scaleadd_scaleadd_relu_fp32.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_fp32 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_fp16
               grouped_convnd_fwd_scaleadd_scaleadd_relu/grouped_conv_fwd_scaleadd_scaleadd_relu_fp16.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_fp16 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_bf16
               grouped_convnd_fwd_scaleadd_scaleadd_relu/grouped_conv_fwd_scaleadd_scaleadd_relu_bf16.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_bf16 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_int8
               grouped_convnd_fwd_scaleadd_scaleadd_relu/grouped_conv_fwd_scaleadd_scaleadd_relu_int8.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_scaleadd_relu_int8 PRIVATE composable_kernel::device_conv_operations)
# Fwd scaleadd AB
add_executable(client_grouped_convnd_fwd_scaleadd_ab_fp32
               grouped_convnd_fwd_scaleadd_ab/grouped_conv_fwd_scaleadd_ab_fp32.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_ab_fp32 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_ab_fp16
               grouped_convnd_fwd_scaleadd_ab/grouped_conv_fwd_scaleadd_ab_fp16.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_ab_fp16 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_ab_bf16
               grouped_convnd_fwd_scaleadd_ab/grouped_conv_fwd_scaleadd_ab_bf16.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_ab_bf16 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_convnd_fwd_scaleadd_ab_int8
               grouped_convnd_fwd_scaleadd_ab/grouped_conv_fwd_scaleadd_ab_int8.cpp)
target_link_libraries(client_grouped_convnd_fwd_scaleadd_ab_int8 PRIVATE composable_kernel::device_conv_operations)
# Fwd bilinear
add_executable(client_grouped_convnd_fwd_bilinear_residual_fp16
               grouped_convnd_fwd_bilinear/grouped_conv_fwd_bilinear_residual_fp16.cpp)
target_link_libraries(client_grouped_convnd_fwd_bilinear_residual_fp16 PRIVATE composable_kernel::device_conv_operations)
# Fwd convinvscale
add_executable(client_conv3d_fwd_convinvscale_fp8
               grouped_convnd_fwd_convinvscale/conv3d_fwd_convinvscale_fp8.cpp)
target_link_libraries(client_conv3d_fwd_convinvscale_fp8 PRIVATE composable_kernel::device_conv_operations)
# Fwd convscale + Bias
add_executable(client_conv3d_fwd_convscale_add_fp8
               grouped_convnd_fwd_convscale_add/conv3d_fwd_convscale_add_fp8.cpp)
target_link_libraries(client_conv3d_fwd_convscale_add_fp8 PRIVATE composable_kernel::device_conv_operations)
# Fwd convscale + ReLU
add_executable(client_conv3d_fwd_convscale_relu_fp8
               grouped_convnd_fwd_convscale_relu/conv3d_fwd_convscale_relu_fp8.cpp)
target_link_libraries(client_conv3d_fwd_convscale_relu_fp8 PRIVATE composable_kernel::device_conv_operations)
# Fwd convscale + ReLU + AMAX
add_executable(client_conv3d_fwd_convscale_relu_amax_fp8
               grouped_convnd_fwd_convscale_reduce/conv3d_fwd_convscale_relu_amax_fp8.cpp)
target_link_libraries(client_conv3d_fwd_convscale_relu_amax_fp8
                      PRIVATE composable_kernel::device_conv_operations
                              composable_kernel::device_other_operations
                              composable_kernel::device_reduction_operations
                              composable_kernel::utility)
# Fwd convscale + AMAX
add_executable(client_conv3d_fwd_convscale_amax_fp8
               grouped_convnd_fwd_convscale_reduce/conv3d_fwd_convscale_amax_fp8.cpp)
target_link_libraries(client_conv3d_fwd_convscale_amax_fp8
                      PRIVATE composable_kernel::device_conv_operations
                              composable_kernel::device_other_operations
                              composable_kernel::device_reduction_operations
                              composable_kernel::utility)
# Fwd convscale
add_executable(client_conv3d_fwd_convscale_fp8
               grouped_convnd_fwd_convscale/conv3d_fwd_convscale_fp8.cpp)
target_link_libraries(client_conv3d_fwd_convscale_fp8 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_conv3d_fwd_convscale_bf8
               grouped_convnd_fwd_convscale/conv3d_fwd_convscale_bf8.cpp)
target_link_libraries(client_conv3d_fwd_convscale_bf8 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_conv3d_fwd_convscale_fp8_bf8
               grouped_convnd_fwd_convscale/conv3d_fwd_convscale_fp8_bf8.cpp)
target_link_libraries(client_conv3d_fwd_convscale_fp8_bf8 PRIVATE composable_kernel::device_conv_operations)

add_executable(client_conv3d_fwd_convscale_bf8_fp8
               grouped_convnd_fwd_convscale/conv3d_fwd_convscale_bf8_fp8.cpp)
target_link_libraries(client_conv3d_fwd_convscale_bf8_fp8 PRIVATE composable_kernel::device_conv_operations)
# Bwd data bilinear
add_executable(client_grouped_convnd_bwd_data_bilinear_residual_fp16
               grouped_convnd_bwd_data_bilinear/grouped_conv_bwd_data_bilinear_residual_fp16.cpp)
target_link_libraries(client_grouped_convnd_bwd_data_bilinear_residual_fp16 PRIVATE composable_kernel::device_conv_operations)
# Bwd weight bilinear
add_executable(client_grouped_convnd_bwd_weight_bilinear_residual_fp16
               grouped_convnd_bwd_weight_bilinear/grouped_conv_bwd_weight_bilinear_residual_fp16.cpp)
target_link_libraries(client_grouped_convnd_bwd_weight_bilinear_residual_fp16 PRIVATE composable_kernel::device_conv_operations)
# Fwd scale
add_executable(client_grouped_convnd_fwd_scale_fp16
               grouped_convnd_fwd_scale/grouped_conv_fwd_scale_fp16.cpp)
target_link_libraries(client_grouped_convnd_fwd_scale_fp16 PRIVATE composable_kernel::device_conv_operations)
# Bwd data scale
add_executable(client_grouped_convnd_bwd_data_scale_fp16
               grouped_convnd_bwd_data_scale/grouped_conv_bwd_data_scale_fp16.cpp)
target_link_libraries(client_grouped_convnd_bwd_data_scale_fp16 PRIVATE composable_kernel::device_conv_operations)
# Bwd weight scale
add_executable(client_grouped_convnd_bwd_weight_scale_fp16
               grouped_convnd_bwd_weight_scale/grouped_conv_bwd_weight_scale_fp16.cpp)
target_link_libraries(client_grouped_convnd_bwd_weight_scale_fp16 PRIVATE composable_kernel::device_conv_operations)
endif()
