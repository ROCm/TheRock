# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_reduce_nhwc_c reduce_nhwc_c.cpp)
target_link_libraries(client_reduce_nhwc_c PRIVATE composable_kernel::device_reduction_operations)
