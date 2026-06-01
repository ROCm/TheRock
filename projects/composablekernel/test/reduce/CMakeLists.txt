# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_reduce_no_index reduce_no_index.cpp)
add_gtest_executable(test_reduce_with_index reduce_with_index.cpp)
target_link_libraries(test_reduce_no_index PRIVATE utility device_reduce_instance)
target_link_libraries(test_reduce_with_index PRIVATE utility device_reduce_instance)

