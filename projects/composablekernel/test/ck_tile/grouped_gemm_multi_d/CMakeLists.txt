# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

# Use standard asm for rtn bf16 conversion instead of turncate
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3)

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_grouped_gemm_multi_d test_grouped_gemm_multi_d.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_multi_d PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()
