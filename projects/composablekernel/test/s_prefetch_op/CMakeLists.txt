# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_test_executable(test_s_prefetch_op s_prefetch_op.cpp)
target_link_libraries(test_s_prefetch_op PRIVATE utility)
