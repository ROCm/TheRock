# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_executable(client_grouped_gemm_fastgelu grouped_gemm_fastgelu.cpp)
    target_link_libraries(client_grouped_gemm_fastgelu PRIVATE composable_kernel::device_gemm_operations)
endif()
