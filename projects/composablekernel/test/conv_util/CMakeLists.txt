# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_conv_util conv_util.cpp)
target_link_libraries(test_conv_util PRIVATE utility)
