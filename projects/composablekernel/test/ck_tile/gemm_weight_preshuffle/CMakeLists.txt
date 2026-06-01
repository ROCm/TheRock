# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile_gemm is only built on gfx94/gfx95
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_gemm_pipeline_wp test_gemm_pipeline_wp.cpp)

    target_compile_options(test_ck_tile_gemm_pipeline_wp PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile_gemm tests for current target")
endif()

