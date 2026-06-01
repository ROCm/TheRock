# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_grouped_conv2d_bwd_data grouped_conv2d_bwd_data.cpp)
target_link_libraries(client_grouped_conv2d_bwd_data PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_conv2d_bwd_data_ngchw grouped_conv2d_bwd_data_ngchw.cpp)
target_link_libraries(client_grouped_conv2d_bwd_data_ngchw PRIVATE composable_kernel::device_conv_operations)

add_executable(client_grouped_conv3d_bwd_data grouped_conv3d_bwd_data.cpp)
target_link_libraries(client_grouped_conv3d_bwd_data PRIVATE composable_kernel::device_conv_operations)

if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "bf8") OR (NOT DEFINED DTYPES AND GPU_TARGETS MATCHES "gfx94"))
    add_executable(client_grouped_conv3d_bwd_data_input_fp16_comp_bf8f8 grouped_conv3d_bwd_data_input_fp16_comp_bf8f8.cpp)
    target_link_libraries(client_grouped_conv3d_bwd_data_input_fp16_comp_bf8f8 PRIVATE composable_kernel::device_conv_operations)
endif()