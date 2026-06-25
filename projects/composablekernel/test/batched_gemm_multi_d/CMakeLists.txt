# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_batched_gemm_multi_d test_batched_gemm_multi_d_dl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_batched_gemm_multi_d PRIVATE utility device_batched_gemm_multi_d_instance)
endif()
