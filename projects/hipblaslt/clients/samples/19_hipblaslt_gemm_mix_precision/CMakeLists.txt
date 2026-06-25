# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_mix_precision "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_mix_precision.cpp")
target_link_libraries(sample_hipblaslt_gemm_mix_precision PRIVATE sample-helper)
