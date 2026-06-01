# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_image_to_column image_to_column.cpp)
target_link_libraries(client_image_to_column PRIVATE composable_kernel::device_other_operations)

add_executable(client_column_to_image column_to_image.cpp)
target_link_libraries(client_column_to_image PRIVATE composable_kernel::device_other_operations)
