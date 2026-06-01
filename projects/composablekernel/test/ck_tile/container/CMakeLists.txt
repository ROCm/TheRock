# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_tuple_apply test_tuple_apply.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_ck_tile_tuple_apply PRIVATE utility)
    endif()
endif()