# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message("-- Adding: test/ck_tile/utility/")

add_gtest_executable(test_fill test_fill.cpp)

# Add print tests
add_subdirectory(print)
