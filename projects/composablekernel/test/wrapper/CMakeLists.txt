# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_wrapper)

add_gtest_executable(test_wrapper_layout test_wrapper_layout.cpp)
target_link_libraries(test_wrapper_layout PRIVATE utility)
add_dependencies(test_wrapper test_wrapper_layout)
add_gtest_executable(test_wrapper_tensor test_wrapper_tensor.cpp)
target_link_libraries(test_wrapper_tensor PRIVATE utility)
add_dependencies(test_wrapper test_wrapper_tensor)
add_gtest_executable(test_wrapper_copy test_wrapper_copy.cpp)
target_link_libraries(test_wrapper_copy PRIVATE utility)
add_dependencies(test_wrapper test_wrapper_copy)
add_gtest_executable(test_wrapper_partition test_wrapper_partition.cpp)
target_link_libraries(test_wrapper_partition PRIVATE utility)
add_dependencies(test_wrapper test_wrapper_partition)
add_gtest_executable(test_wrapper_gemm test_wrapper_gemm_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_wrapper_gemm PRIVATE utility)
    add_dependencies(test_wrapper test_wrapper_gemm)
endif()
