# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_normalization_bwd_data)

add_gtest_executable(test_layernorm2d_bwd_data_fp32 test_layernorm2d_bwd_data_fp32.cpp)
if (result EQUAL 0)
    target_link_libraries(test_layernorm2d_bwd_data_fp32 PRIVATE utility device_normalization_bwd_data_instance)
    add_dependencies(test_normalization_bwd_data test_layernorm2d_bwd_data_fp32)
endif()

add_gtest_executable(test_groupnorm_bwd_data_fp32 test_groupnorm_bwd_data_fp32.cpp)
if (result EQUAL 0)
    target_link_libraries(test_groupnorm_bwd_data_fp32 PRIVATE utility device_normalization_bwd_data_instance)
    add_dependencies(test_normalization_bwd_data test_groupnorm_bwd_data_fp32)
endif()
