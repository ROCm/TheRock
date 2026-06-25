# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

add_subdirectory(01_fmha)
add_subdirectory(02_layernorm2d)
add_subdirectory(03_gemm)
add_subdirectory(04_img2col)
add_subdirectory(05_reduce)
add_subdirectory(06_permute)
add_subdirectory(09_topk_softmax)
add_subdirectory(10_rmsnorm2d)
add_subdirectory(11_add_rmsnorm2d_rdquant)
add_subdirectory(12_smoothquant)
add_subdirectory(13_moe_sorting)
add_subdirectory(14_moe_smoothquant)
add_subdirectory(15_fused_moe)
add_subdirectory(16_batched_gemm)
add_subdirectory(17_grouped_gemm)
add_subdirectory(18_flatmm)
add_subdirectory(19_gemm_multi_d)
add_subdirectory(20_grouped_convolution)
add_subdirectory(21_elementwise)
add_subdirectory(22_gemm_multi_abd)
add_subdirectory(35_batched_transpose)
add_subdirectory(36_pooling)
add_subdirectory(38_block_scale_gemm)
add_subdirectory(40_streamk_gemm)
add_subdirectory(41_batched_contraction)

