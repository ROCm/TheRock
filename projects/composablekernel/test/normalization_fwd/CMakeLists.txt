# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_normalization_fwd)
add_gtest_executable(test_layernorm2d_fwd_fp32 test_layernorm2d_fwd_fp32.cpp)
if(result EQUAL 0)
  target_link_libraries(test_layernorm2d_fwd_fp32 PRIVATE utility device_normalization_fwd_instance)
  add_dependencies(test_normalization_fwd test_layernorm2d_fwd_fp32)
endif()

add_gtest_executable(test_groupnorm_fwd_fp32 test_groupnorm_fwd_fp32.cpp)
if(result EQUAL 0)
  target_link_libraries(test_groupnorm_fwd_fp32 PRIVATE utility device_normalization_fwd_instance)
  add_dependencies(test_normalization_fwd test_groupnorm_fwd_fp32)
endif()

add_gtest_executable(test_layernorm2d_fwd_fp16 test_layernorm2d_fwd_fp16.cpp)
if(result EQUAL 0)
  target_link_libraries(test_layernorm2d_fwd_fp16 PRIVATE utility device_normalization_fwd_instance)
  add_dependencies(test_normalization_fwd test_layernorm2d_fwd_fp16)
endif()

add_gtest_executable(test_layernorm4d_fwd_fp16 test_layernorm4d_fwd_fp16.cpp)
if(result EQUAL 0)
  target_link_libraries(test_layernorm4d_fwd_fp16 PRIVATE utility device_normalization_fwd_instance)
  add_dependencies(test_normalization_fwd test_layernorm4d_fwd_fp16)
endif()

add_gtest_executable(test_groupnorm_fwd_fp16 test_groupnorm_fwd_fp16.cpp)
if(result EQUAL 0)
  target_link_libraries(test_groupnorm_fwd_fp16 PRIVATE utility device_normalization_fwd_instance)
  add_dependencies(test_normalization_fwd test_groupnorm_fwd_fp16)
endif()
