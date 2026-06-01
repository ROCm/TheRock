# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
  add_gtest_executable(test_gemm_add_relu_add_layernorm_fp16 test_gemm_add_relu_add_layernorm_fp16.cpp)
  if(result EQUAL 0)
    add_custom_target(test_gemm_layernorm)
    target_link_libraries(test_gemm_add_relu_add_layernorm_fp16 PRIVATE utility device_gemm_add_relu_add_layernorm_instance)
    add_dependencies(test_gemm_layernorm test_gemm_add_relu_add_layernorm_fp16)
  endif()
endif()
