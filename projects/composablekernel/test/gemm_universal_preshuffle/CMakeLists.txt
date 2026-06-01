# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9[45]|gfx12")
    add_gtest_executable(test_gemm_universal_preshuffle_xdl_fp8 test_gemm_universal_preshuffle_xdl_fp8.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_universal_preshuffle_xdl_fp8 PRIVATE utility device_gemm_universal_preshuffle_instance)
    endif()
endif()
