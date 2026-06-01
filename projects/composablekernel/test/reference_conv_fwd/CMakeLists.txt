# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_reference_conv_fwd reference_conv_fwd.cpp)
target_link_libraries(test_reference_conv_fwd PRIVATE utility)
