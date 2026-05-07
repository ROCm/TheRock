# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_softmax4d softmax4d.cpp)
target_link_libraries(client_softmax4d PRIVATE composable_kernel::device_other_operations composable_kernel::device_reduction_operations)
