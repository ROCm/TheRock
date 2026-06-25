# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(tile_tutorial_naive_gemm practice_gemm.cpp)

target_compile_options(tile_tutorial_naive_gemm PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)

add_dependencies(tutorials tile_tutorial_naive_gemm)