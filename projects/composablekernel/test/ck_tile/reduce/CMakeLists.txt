# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_reduce2d test_reduce2d.cpp)
    add_gtest_executable(test_ck_tile_multi_reduce2d_threadwise test_multi_reduce2d_threadwise.cpp)
    add_gtest_executable(test_ck_tile_multi_reduce2d_multiblock test_multi_reduce2d_multiblock.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_ck_tile_reduce2d PRIVATE utility)
        target_link_libraries(test_ck_tile_multi_reduce2d_threadwise PRIVATE utility)
        target_link_libraries(test_ck_tile_multi_reduce2d_multiblock PRIVATE utility)
    endif()
endif()

