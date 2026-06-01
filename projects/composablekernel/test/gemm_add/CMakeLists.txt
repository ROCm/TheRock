# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Implements test instances for MultipleD with xdl and wmma support.

add_gtest_executable(test_gemm_add_xdl test_gemm_add_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_xdl PRIVATE utility device_gemm_add_instance)
endif()

add_gtest_executable(test_gemm_add_relu_xdl test_gemm_add_relu_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_relu_xdl PRIVATE utility device_gemm_add_instance device_gemm_add_relu_instance)
endif()

add_gtest_executable(test_gemm_add_silu_xdl test_gemm_add_silu_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_silu_xdl PRIVATE utility device_gemm_add_instance device_gemm_add_silu_instance)
endif()

add_gtest_executable(test_gemm_add_silu_wmma test_gemm_add_silu_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_silu_wmma PRIVATE utility device_gemm_add_instance device_gemm_add_silu_instance)
endif()

add_gtest_executable(test_gemm_add_fastgelu_xdl test_gemm_add_fastgelu_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_fastgelu_xdl PRIVATE utility device_gemm_add_instance device_gemm_add_fastgelu_instance)
endif()

add_gtest_executable(test_gemm_fastgelu_wmma test_gemm_fastgelu_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_fastgelu_wmma PRIVATE utility device_gemm_fastgelu_instance)
endif()

add_gtest_executable(test_gemm_add_fastgelu_wmma test_gemm_add_fastgelu_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_fastgelu_wmma PRIVATE utility device_gemm_add_fastgelu_instance)
endif()

add_gtest_executable(test_gemm_add_wmma test_gemm_add_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_wmma PRIVATE utility device_gemm_add_instance)
endif()

add_gtest_executable(test_gemm_add_add_fastgelu_wmma test_gemm_add_add_fastgelu_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_add_fastgelu_wmma PRIVATE utility device_gemm_add_add_fastgelu_instance)
endif()

add_gtest_executable(test_gemm_multiply_multiply_wmma test_gemm_multiply_multiply_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_multiply_multiply_wmma PRIVATE utility device_gemm_multiply_multiply_instance)
endif()

add_gtest_executable(test_gemm_add_multiply_wmma test_gemm_add_multiply_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_multiply_wmma PRIVATE utility device_gemm_add_multiply_instance)
endif()

add_gtest_executable(test_gemm_multiply_add_wmma test_gemm_multiply_add_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_multiply_add_wmma PRIVATE utility device_gemm_multiply_add_instance)
endif()

add_gtest_executable(test_gemm_bilinear_wmma test_gemm_bilinear_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_bilinear_wmma PRIVATE utility device_gemm_bilinear_instance)
endif()

add_gtest_executable(test_gemm_add_relu_wmma test_gemm_add_relu_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_relu_wmma PRIVATE utility device_gemm_add_relu_instance)
endif()