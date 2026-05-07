# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(example_fpAintB_gemm_wmma)
add_example_executable(example_fp16int8_gemm_wmma fp16int8_gemm_wmma.cpp)
add_example_dependencies(example_fpAintB_gemm_wmma example_fp16int8_gemm_wmma)
