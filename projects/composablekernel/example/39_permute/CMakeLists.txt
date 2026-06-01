# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(example_permute)

add_example_executable(example_permute_1xHxW_fp16 permute_1xHxW_fp16.cpp)
add_example_dependencies(example_permute example_permute_1xHxW_fp16)

add_example_executable(example_permute_NxHxW_fp16 permute_NxHxW_fp16.cpp)
add_example_dependencies(example_permute example_permute_NxHxW_fp16)

add_example_executable(example_permute_HxWx4_fp16 permute_HxWx4_fp16.cpp)
add_example_dependencies(example_permute example_permute_HxWx4_fp16)
