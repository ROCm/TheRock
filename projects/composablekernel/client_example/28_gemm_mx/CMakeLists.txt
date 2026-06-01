# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx950")
	add_executable(client_gemm_mx_fp8 gemm_mx_fp8.cpp)
	target_link_libraries(client_gemm_mx_fp8 PRIVATE composable_kernel::device_gemm_operations)
endif()
