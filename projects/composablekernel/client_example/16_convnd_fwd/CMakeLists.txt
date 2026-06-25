# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if((DTYPES MATCHES "fp16") OR NOT DEFINED DTYPES)
    add_executable(client_conv3d_fwd_fp16 conv3d_fwd_fp16.cpp)
    target_link_libraries(client_conv3d_fwd_fp16 PRIVATE composable_kernel::device_conv_operations)

endif()

if((DTYPES MATCHES "fp8") OR (NOT DEFINED DTYPES AND GPU_TARGETS MATCHES "gfx94"))
    add_executable(client_conv3d_fwd_fp16_comp_fp8 conv3d_fwd_fp16_comp_fp8.cpp)
    target_link_libraries(client_conv3d_fwd_fp16_comp_fp8 PRIVATE composable_kernel::device_conv_operations)
endif()

if((DTYPES MATCHES "fp32") OR NOT DEFINED DTYPES)
    add_executable(client_conv3d_fwd_fp32 conv3d_fwd_fp32.cpp)
    target_link_libraries(client_conv3d_fwd_fp32 PRIVATE composable_kernel::device_conv_operations)
endif()
