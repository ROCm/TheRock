# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_gemm_add_multiply gemm_add_multiply.cpp)
    target_link_libraries(client_gemm_add_multiply PRIVATE composable_kernel::device_gemm_operations)
endif()
