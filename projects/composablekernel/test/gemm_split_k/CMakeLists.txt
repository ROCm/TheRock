# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gemm_splitk test_gemm_splitk_xdl.cpp)
if(result EQUAL 0)
   target_link_libraries(test_gemm_splitk PRIVATE utility device_gemm_splitk_instance)
 endif()
