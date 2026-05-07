# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(tile_tutorial_copy_kernel copy_basic.cpp)

# Impact: This flag ensures that the compiler doesn't make 
# assumptions about memory aliasing that could interfere with Composable Kernel's explicit memory access patterns.
target_compile_options(tile_tutorial_copy_kernel PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)

add_dependencies(tutorials tile_tutorial_copy_kernel)
