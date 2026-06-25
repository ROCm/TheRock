# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(example_gemm_add_xdl)

add_example_executable(example_gemm_add_xdl_fp16 gemm_add_xdl_fp16.cpp)
add_example_dependencies(example_gemm_add_xdl example_gemm_add_xdl_fp16)


add_example_executable(example_gemm_add_xdl_bf16 gemm_add_xdl_bf16.cpp)
add_example_dependencies(example_gemm_add_xdl example_gemm_add_xdl_bf16)

add_custom_target(example_gemm_add_wmma)

add_example_executable(example_gemm_add_wmma_bf16 gemm_add_wmma_bf16.cpp)
add_example_dependencies(example_gemm_add_wmma example_gemm_add_wmma_bf16)

add_example_executable(example_gemm_add_wmma_fp16 gemm_add_wmma_fp16.cpp)
add_example_dependencies(example_gemm_add_wmma example_gemm_add_wmma_fp16)






