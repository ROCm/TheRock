# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_elementwise_normalization)
add_gtest_executable(test_elementwise_layernorm_fp16 test_elementwise_layernorm_fp16.cpp)
if(result EQUAL 0)
  target_link_libraries(test_elementwise_layernorm_fp16 PRIVATE utility device_elementwise_normalization_instance)
  add_dependencies(test_elementwise_normalization test_elementwise_layernorm_fp16)
endif()