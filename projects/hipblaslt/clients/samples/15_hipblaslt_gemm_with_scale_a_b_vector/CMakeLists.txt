# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_with_scale_a_b_vector "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_with_scale_a_b_vector.cpp")
target_link_libraries(sample_hipblaslt_gemm_with_scale_a_b_vector PRIVATE sample-helper)

add_executable(sample_hipblaslt_gemm_swizzleA_with_scale_a_b_vector "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_swizzleA_with_scale_a_b_vector.cpp")
target_link_libraries(sample_hipblaslt_gemm_swizzleA_with_scale_a_b_vector PRIVATE sample-helper)
