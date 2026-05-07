# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_grouped_conv2d_fwd grouped_conv2d_fwd.cpp)
    target_link_libraries(client_grouped_conv2d_fwd PRIVATE composable_kernel::device_conv_operations)

    add_executable(client_grouped_conv1d_fwd grouped_conv1d_fwd.cpp)
    target_link_libraries(client_grouped_conv1d_fwd PRIVATE composable_kernel::device_conv_operations)

    if((DTYPES MATCHES "fp8") OR (NOT DEFINED DTYPES AND GPU_TARGETS MATCHES "gfx94"))
        add_executable(client_grouped_conv3d_fwd_fp8 grouped_conv3d_fwd_fp8.cpp)
        target_link_libraries(client_grouped_conv3d_fwd_fp8 PRIVATE composable_kernel::device_conv_operations)
    endif()

    if((DTYPES MATCHES "bf8") OR (NOT DEFINED DTYPES AND GPU_TARGETS MATCHES "gfx94"))
        add_executable(client_grouped_conv3d_fwd_bf8 grouped_conv3d_fwd_bf8.cpp)
        target_link_libraries(client_grouped_conv3d_fwd_bf8 PRIVATE composable_kernel::device_conv_operations)
    endif()

    if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "bf8") OR (NOT DEFINED DTYPES AND GPU_TARGETS MATCHES "gfx94"))
        add_executable(client_grouped_conv3d_fwd_fp8_bf8 grouped_conv3d_fwd_fp8_bf8.cpp)
        target_link_libraries(client_grouped_conv3d_fwd_fp8_bf8 PRIVATE composable_kernel::device_conv_operations)

        add_executable(client_grouped_conv3d_fwd_bf8_fp8 grouped_conv3d_fwd_bf8_fp8.cpp)
        target_link_libraries(client_grouped_conv3d_fwd_bf8_fp8 PRIVATE composable_kernel::device_conv_operations)
    endif()

    add_executable(grouped_conv2d_fwd_ngchw grouped_conv2d_fwd_ngchw.cpp)
    target_link_libraries(grouped_conv2d_fwd_ngchw PRIVATE composable_kernel::device_conv_operations)
endif()
