# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_executable(tile_example_streamk_gemm_basic streamk_gemm_basic.cpp)
    set(EXAMPLE_GEMM_COMPILE_OPTIONS)
    if(CK_USE_OCP_FP8)
	    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
    endif()
    target_compile_options(tile_example_streamk_gemm_basic PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile streamk gemm tests for current target")
endif()
