# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_max_pool2d_fwd max_pool2d_fwd.cpp)
target_link_libraries(client_max_pool2d_fwd PRIVATE composable_kernel::device_other_operations)

add_executable(client_max_pool2d_bwd max_pool2d_bwd.cpp)
target_link_libraries(client_max_pool2d_bwd PRIVATE composable_kernel::device_other_operations)

add_executable(client_avg_pool3d_fwd avg_pool3d_fwd.cpp)
target_link_libraries(client_avg_pool3d_fwd PRIVATE composable_kernel::device_other_operations)

add_executable(client_avg_pool3d_bwd avg_pool3d_bwd.cpp)
target_link_libraries(client_avg_pool3d_bwd PRIVATE composable_kernel::device_other_operations)
