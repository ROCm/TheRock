# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx94|gfx95")
    add_gtest_executable(test_ck_tile_grouped_gemm_abquant_1x1x128 test_grouped_gemm_abquant_1x1x128.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_abquant_1x1x128 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_ck_tile_grouped_gemm_abquant_1x128x128 test_grouped_gemm_abquant_1x128x128.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_abquant_1x128x128 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

