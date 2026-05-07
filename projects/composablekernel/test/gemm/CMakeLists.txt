# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_test_executable(test_gemm_fp32 gemm_fp32.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_fp32 PRIVATE utility device_gemm_instance)
endif()
add_test_executable(test_gemm_fp16 gemm_fp16.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_fp16 PRIVATE utility device_gemm_instance)
    add_library(gemm_standalone_xdl_fp16_instances STATIC
    instance/gemm_f16_nn_instance.cpp
    instance/gemm_f16_nt_instance.cpp
    instance/gemm_f16_tn_instance.cpp
    instance/gemm_wavelet_f16_tn_instance.cpp
    instance/gemm_f16_tt_instance.cpp
    )
endif()
add_test_executable(test_gemm_standalone_xdl_fp16 gemm_standalone_xdl_fp16.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_standalone_xdl_fp16 PRIVATE gemm_standalone_xdl_fp16_instances utility)
    target_include_directories(test_gemm_standalone_xdl_fp16 PRIVATE instance/)
endif()
add_test_executable(test_gemm_bf16 gemm_bf16.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_bf16 PRIVATE utility device_gemm_instance)
endif()
add_test_executable(test_gemm_int8 gemm_int8.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_int8 PRIVATE utility device_gemm_instance)
endif()