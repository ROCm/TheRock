# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_elementwise_transpose3d elementwise_transpose_3d.cpp)
target_link_libraries(client_elementwise_transpose3d PRIVATE composable_kernel::device_other_operations)
