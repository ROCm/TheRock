# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Print utility tests
add_gtest_executable(test_print_sequence test_print_sequence.cpp)
add_gtest_executable(test_print_array test_print_array.cpp)
add_gtest_executable(test_print_tuple test_print_tuple.cpp)
add_gtest_executable(test_print_coordinate_transform test_print_coordinate_transform.cpp)
add_gtest_executable(test_print_static_encoding_pattern test_print_static_encoding_pattern.cpp)
add_gtest_executable(test_print_buffer_view test_print_buffer_view.cpp)
add_gtest_executable(test_print_basic_types test_print_basic_types.cpp)
add_gtest_executable(test_print_tile_window test_print_tile_window.cpp)

# Apply OCP FP8 flag for tile_window test to ensure host/device FP8 format consistency
if(CK_USE_OCP_FP8)
    target_compile_options(test_print_tile_window PRIVATE -DCK_TILE_USE_OCP_FP8)
endif()
