# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_permute)
add_gtest_executable(test_permute_scale test_permute_scale.cpp)
target_link_libraries(test_permute_scale PRIVATE utility device_permute_scale_instance)
add_dependencies(test_permute test_permute_scale)
