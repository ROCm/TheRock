# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_gemm gemm.cpp)
target_link_libraries(client_gemm PRIVATE composable_kernel::device_other_operations composable_kernel::device_gemm_operations)
