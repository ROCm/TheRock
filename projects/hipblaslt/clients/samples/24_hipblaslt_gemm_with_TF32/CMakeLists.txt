# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_with_TF32 "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_with_TF32.cpp")
target_link_libraries(sample_hipblaslt_gemm_with_TF32 PRIVATE sample-helper)
