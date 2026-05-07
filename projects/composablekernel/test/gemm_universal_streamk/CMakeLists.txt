# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gemm_universal_streamk_fp16 test_gemm_universal_streamk_xdl_fp16.cpp)
if(result EQUAL 0)
   target_link_libraries(test_gemm_universal_streamk_fp16 PRIVATE utility device_gemm_universal_streamk_instance)
endif()

add_gtest_executable(test_gemm_universal_streamk_fp8 test_gemm_universal_streamk_xdl_fp8.cpp)
if(result EQUAL 0)
   target_link_libraries(test_gemm_universal_streamk_fp8 PRIVATE utility device_gemm_universal_streamk_instance)
endif()

add_gtest_executable(test_gemm_universal_streamk_bf16 test_gemm_universal_streamk_xdl_bf16.cpp)
if(result EQUAL 0)
   target_link_libraries(test_gemm_universal_streamk_bf16 PRIVATE utility device_gemm_universal_streamk_instance)
endif()

