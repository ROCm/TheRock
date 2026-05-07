# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_batchnorm_fwd_nhwc batchnorm_fwd_nhwc.cpp)
add_executable(client_batchnorm_bwd_nhwc batchnorm_bwd_nhwc.cpp)
add_executable(client_batchnorm_infer_nhwc batchnorm_infer_nhwc.cpp)
target_link_libraries(client_batchnorm_fwd_nhwc PRIVATE composable_kernel::device_other_operations)
target_link_libraries(client_batchnorm_bwd_nhwc PRIVATE composable_kernel::device_other_operations)
target_link_libraries(client_batchnorm_infer_nhwc PRIVATE composable_kernel::device_other_operations)
