# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_test_executable(test_wmma_op wmma_op.cpp)
target_link_libraries(test_wmma_op PRIVATE utility)
