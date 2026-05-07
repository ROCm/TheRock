# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_gemm_bias_add_reduce_fp16 test_gemm_bias_add_reduce_fp16.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_bias_add_reduce_fp16 PRIVATE utility device_gemm_bias_add_reduce_instance)
    endif()
endif()
