# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
  add_test_executable(test_gemm_reduce_fp16 gemm_reduce_fp16.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_gemm_reduce_fp16 PRIVATE utility device_gemm_reduce_instance)
  endif()
endif()
