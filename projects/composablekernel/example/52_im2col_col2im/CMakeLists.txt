# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(example_im2col_col2im)

add_example_executable(example_image_to_column_f32 image_to_column_f32.cpp)
add_example_dependencies(example_im2col_col2im example_image_to_column_f32)

add_example_executable(example_column_to_image_f32 column_to_image_f32.cpp)
add_example_dependencies(example_im2col_col2im example_column_to_image_f32)
