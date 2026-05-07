# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_tensor_transform_using_wrapper tensor_transform_using_wrapper.cpp)
target_link_libraries(client_tensor_transform_using_wrapper PRIVATE composable_kernel::device_other_operations)
add_executable(client_wrapper_img2col wrapper_img2col.cpp)
target_link_libraries(client_wrapper_img2col PRIVATE composable_kernel::device_other_operations)
if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_wrapper_basic_gemm wrapper_basic_gemm.cpp)
    target_link_libraries(client_wrapper_basic_gemm PRIVATE composable_kernel::device_other_operations)
    add_executable(client_wrapper_optimized_gemm wrapper_optimized_gemm.cpp)
    target_link_libraries(client_wrapper_optimized_gemm PRIVATE composable_kernel::device_other_operations)
endif()
