# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_swizzle_b "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_swizzle_b.cpp")
target_link_libraries(sample_hipblaslt_gemm_swizzle_b PRIVATE sample-helper)
