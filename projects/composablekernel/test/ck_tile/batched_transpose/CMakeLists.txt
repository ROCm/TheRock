# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_batched_transpose test_batched_transpose.cpp)
    set_property(TARGET test_ck_tile_batched_transpose PROPERTY CXX_STANDARD 20)
else()
    message(DEBUG "Skipping ck_tile batched_transpose tests for current target")
endif()
