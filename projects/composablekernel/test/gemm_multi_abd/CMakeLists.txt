# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gemm_multi_abd_wmma test_gemm_multi_abd_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_multi_abd_wmma PRIVATE utility device_gemm_multi_abd_instance)
endif()

add_gtest_executable(test_gemm_multi_abd_xdl test_gemm_multi_abd_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_multi_abd_xdl PRIVATE utility device_gemm_multi_abd_instance)
endif()
