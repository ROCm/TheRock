# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_batched_gemm_xdl test_batched_gemm_xdl.cpp)
if(result EQUAL 0)
   target_link_libraries(test_batched_gemm_xdl PRIVATE utility device_batched_gemm_instance)
endif()

add_gtest_executable(test_batched_gemm_wmma test_batched_gemm_wmma.cpp)
if(result EQUAL 0)
   target_link_libraries(test_batched_gemm_wmma PRIVATE utility device_batched_gemm_instance)
endif()
