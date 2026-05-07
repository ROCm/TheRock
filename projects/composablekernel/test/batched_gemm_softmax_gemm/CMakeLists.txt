# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_batched_gemm_softmax_gemm_fp16 test_batched_gemm_softmax_gemm_fp16_xdl.cpp)
if(result EQUAL 0)
  add_custom_target(test_batched_gemm_softmax_gemm)
  target_link_libraries(test_batched_gemm_softmax_gemm_fp16 PRIVATE utility device_batched_gemm_softmax_gemm_instance)
  add_dependencies(test_batched_gemm_softmax_gemm test_batched_gemm_softmax_gemm_fp16)
endif()
