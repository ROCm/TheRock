# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(device_mem_tests)
add_gtest_executable(test_device_prng test_device_prng.cpp)
target_link_libraries(test_device_prng PRIVATE utility)
add_dependencies(test_device_prng device_mem_tests)
