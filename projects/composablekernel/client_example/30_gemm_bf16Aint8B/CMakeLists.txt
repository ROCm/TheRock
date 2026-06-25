# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9" AND ((DTYPES MATCHES "int8" AND DTYPES MATCHES "bf16") OR NOT DEFINED DTYPES))
	add_executable(client_gemm_bias_fastgelu_bf16_i8_bf16 gemm_bias_fastgelu_xdl_bf16_i8.cpp)
	target_link_libraries(client_gemm_bias_fastgelu_bf16_i8_bf16 PRIVATE composable_kernel::device_gemm_operations)

	add_executable(client_gemm_bias_bf16_i8_bf16 gemm_bias_xdl_bf16_i8.cpp)
	target_link_libraries(client_gemm_bias_bf16_i8_bf16 PRIVATE composable_kernel::device_gemm_operations)

	add_executable(client_gemm_gelu_bf16_i8_bf16 gemm_xdl_gelu_bf16_i8.cpp)
	target_link_libraries(client_gemm_gelu_bf16_i8_bf16 PRIVATE composable_kernel::device_gemm_operations)

	add_executable(client_gemm_bf16_i8_bf16 gemm_xdl_bf16_i8.cpp)
	target_link_libraries(client_gemm_bf16_i8_bf16 PRIVATE composable_kernel::device_gemm_operations)

	add_executable(client_gemm_multiply_bf16_i8_bf16 gemm_xdl_multiply_bf16_i8.cpp)
	target_link_libraries(client_gemm_multiply_bf16_i8_bf16 PRIVATE composable_kernel::device_gemm_operations)
endif()
