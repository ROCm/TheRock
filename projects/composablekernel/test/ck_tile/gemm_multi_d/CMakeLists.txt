# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx12")
    add_gtest_executable(test_gemm_multi_d_cshuffle test_gemm_multi_d_cshuffle.cpp)
    add_gtest_executable(test_gemm_multi_d_default2d test_gemm_multi_d_default2d.cpp)
    target_compile_definitions(test_gemm_multi_d_cshuffle PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_definitions(test_gemm_multi_d_default2d PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()
