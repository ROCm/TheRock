# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gemm_b_scale_xdl test_gemm_b_scale_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_b_scale_xdl PRIVATE utility device_gemm_b_scale_instance)
endif()

add_gtest_executable(test_gemm_b_scale_wmma test_gemm_b_scale_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_b_scale_wmma PRIVATE utility device_gemm_b_scale_instance)
endif()
