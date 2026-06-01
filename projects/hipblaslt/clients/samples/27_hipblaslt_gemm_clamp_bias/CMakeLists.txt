# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_clamp_bias "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_clamp_bias.cpp")
target_link_libraries(sample_hipblaslt_gemm_clamp_bias PRIVATE sample-helper)
