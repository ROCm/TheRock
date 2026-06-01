# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_batched_gemm_gemm_fp16_xdl test_batched_gemm_gemm_fp16_xdl.cpp)
if(result EQUAL 0)
  target_link_libraries(test_batched_gemm_gemm_fp16_xdl PRIVATE utility device_batched_gemm_gemm_instance)
endif()

add_gtest_executable(test_batched_gemm_gemm_bf16_wmma test_batched_gemm_gemm_bf16_wmma_cshuffle_v3.cpp)
if(result EQUAL 0)
  target_link_libraries(test_batched_gemm_gemm_bf16_wmma PRIVATE utility device_batched_gemm_gemm_instance)
endif()

add_gtest_executable(test_batched_gemm_gemm_fp16_wmma test_batched_gemm_gemm_fp16_wmma_cshuffle_v3.cpp)
if(result EQUAL 0)
  target_link_libraries(test_batched_gemm_gemm_fp16_wmma PRIVATE utility device_batched_gemm_gemm_instance)
endif()
