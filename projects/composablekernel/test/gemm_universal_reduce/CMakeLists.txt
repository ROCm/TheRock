# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gemm_universal_reduce_bf16_wmma test_gemm_universal_reduce_bf16_wmma.cpp)
if(result EQUAL 0)
  target_link_libraries(test_gemm_universal_reduce_bf16_wmma PRIVATE utility device_gemm_universal_reduce_instance)
endif()

add_gtest_executable(test_gemm_universal_reduce_fp16_wmma test_gemm_universal_reduce_fp16_wmma.cpp)
if(result EQUAL 0)
  target_link_libraries(test_gemm_universal_reduce_fp16_wmma PRIVATE utility device_gemm_universal_reduce_instance)
endif()

add_gtest_executable(test_gemm_universal_reduce_bf16A_i8_wmma test_gemm_universal_reduce_bf16A_i8_wmma.cpp)
if(result EQUAL 0)
  target_link_libraries(test_gemm_universal_reduce_bf16A_i8_wmma PRIVATE utility device_gemm_universal_reduce_instance)
endif()
