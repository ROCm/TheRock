# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gemm_mx test_gemm_mx.cpp)
if(result EQUAL 0)
   target_compile_options(test_gemm_mx PRIVATE -mavx512f)
   target_link_libraries(test_gemm_mx PRIVATE utility device_gemm_mx_instance)
 endif()
