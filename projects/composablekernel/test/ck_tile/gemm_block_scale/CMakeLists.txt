# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEST_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND TEST_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND TEST_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx12")
    # Typed Test Suite for GEMM Quantization - split into multiple files to reduce compile time
    
    # AQuant tests - split into 6 files
    add_gtest_executable(test_tile_gemm_quant_aquant_base_rcr 
        test_gemm_quant_aquant_base_rcr.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_base_rcr PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_base_rrr_crr 
        test_gemm_quant_aquant_base_rrr_crr.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_base_rrr_crr PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_base_ccr 
        test_gemm_quant_aquant_base_ccr.cpp
    )

    target_compile_options(test_tile_gemm_quant_aquant_base_ccr PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # ABQuant tests
    add_gtest_executable(test_tile_gemm_quant_abquant_base
        test_gemm_quant_abquant_base.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_base PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_padding
        test_gemm_quant_abquant_padding.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_padding PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_preshuffle
        test_gemm_quant_abquant_preshuffle_2d.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_preshuffle PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # AQuant tests
    add_gtest_executable(test_tile_gemm_quant_aquant_prefill 
        test_gemm_quant_aquant_prefill.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_prefill PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_transpose_c 
        test_gemm_quant_aquant_transpose_c.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_transpose_c PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_preshuffle 
        test_gemm_quant_aquant_preshuffle.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_preshuffle PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # BQuant tests (without PreshuffleB) - split into 6 files
    add_gtest_executable(test_tile_gemm_quant_bquant_1d_128 
        test_gemm_quant_bquant_1d_128.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_1d_128 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_1d_64 
        test_gemm_quant_bquant_1d_64.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_1d_64 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_2d_small_n 
        test_gemm_quant_bquant_2d_small_n.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_2d_small_n PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_2d_medium_n 
        test_gemm_quant_bquant_2d_medium_n.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_2d_medium_n PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_2d_large_n 
        test_gemm_quant_bquant_2d_large_n.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_2d_large_n PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_transpose 
        test_gemm_quant_bquant_transpose.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_transpose PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # BQuant tests (with PreshuffleB) - split into 5 files
    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_decode_1d 
        test_gemm_quant_bquant_preshuffle_decode_1d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_decode_1d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_prefill_1d 
        test_gemm_quant_bquant_preshuffle_prefill_1d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_prefill_1d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_tiled_permute 
        test_gemm_quant_bquant_preshuffle_tiled_permute.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_tiled_permute PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_decode_2d 
        test_gemm_quant_bquant_preshuffle_decode_2d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_decode_2d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_prefill_2d 
        test_gemm_quant_bquant_preshuffle_prefill_2d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_prefill_2d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # RowColQuant tests
    add_gtest_executable(test_tile_gemm_quant_rowcol 
        test_gemm_quant_rowcol.cpp
    )
    target_compile_options(test_tile_gemm_quant_rowcol PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # TensorQuant tests
    add_gtest_executable(test_tile_gemm_quant_tensor 
        test_gemm_quant_tensor.cpp
    )
    target_compile_options(test_tile_gemm_quant_tensor PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # Umbrella target to build all gemm quant tests
    add_custom_target(test_tile_gemm_quant_all)
    add_dependencies(test_tile_gemm_quant_all
        # AQuant tests
        test_tile_gemm_quant_aquant_base_rcr
        test_tile_gemm_quant_aquant_base_rrr_crr
        test_tile_gemm_quant_aquant_base_ccr
        test_tile_gemm_quant_aquant_prefill
        test_tile_gemm_quant_aquant_transpose_c
        test_tile_gemm_quant_aquant_preshuffle
        # BQuant tests
        test_tile_gemm_quant_bquant_1d_128
        test_tile_gemm_quant_bquant_1d_64
        test_tile_gemm_quant_bquant_2d_small_n
        test_tile_gemm_quant_bquant_2d_medium_n
        test_tile_gemm_quant_bquant_2d_large_n
        test_tile_gemm_quant_bquant_transpose
        # BQuant preshuffle tests
        test_tile_gemm_quant_bquant_preshuffle_decode_1d
        test_tile_gemm_quant_bquant_preshuffle_prefill_1d
        test_tile_gemm_quant_bquant_preshuffle_tiled_permute
        test_tile_gemm_quant_bquant_preshuffle_decode_2d
        test_tile_gemm_quant_bquant_preshuffle_prefill_2d
        # Other quant tests
        test_tile_gemm_quant_rowcol
        test_tile_gemm_quant_tensor
    )
else()
    message(DEBUG "Skipping ck_tile quant gemm tests for current target")
endif()
