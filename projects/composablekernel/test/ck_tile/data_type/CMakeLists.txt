# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_pk_int4 test_pk_int4.cpp)
endif()
if(GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_pk_fp4 test_pk_fp4.cpp)
    add_gtest_executable(test_ck_tile_mx_scale test_mx_scale.cpp)
endif()

if(CK_USE_OCP_FP8 OR CK_USE_FNUZ_FP8)
    add_gtest_executable(test_ck_tile_fp8 test_fp8.cpp)
    target_compile_options(test_ck_tile_fp8 PRIVATE -Wno-float-equal)
    target_compile_definitions(test_ck_tile_fp8 PUBLIC GTEST_HAS_RTTI=0)
    # conditionally specify the use of OCP_FP8
    if(CK_USE_OCP_FP8)
        target_compile_options(test_ck_tile_fp8 PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()
endif()
