# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(tile_example_gemm_multi_d_fp16 gemm_multi_d_fp16.cpp)
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
target_compile_options(tile_example_gemm_multi_d_fp16 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
