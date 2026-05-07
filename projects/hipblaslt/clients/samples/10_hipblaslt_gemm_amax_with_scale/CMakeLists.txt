# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_amax_with_scale "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_amax_with_scale.cpp")
target_link_libraries(sample_hipblaslt_gemm_amax_with_scale PRIVATE sample-helper)
