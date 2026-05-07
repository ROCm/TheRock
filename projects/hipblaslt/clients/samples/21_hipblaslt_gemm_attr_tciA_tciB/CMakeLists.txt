# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_attr_tciA_tciB "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_attr_tciA_tciB.cpp")
target_link_libraries(sample_hipblaslt_gemm_attr_tciA_tciB PRIVATE sample-helper)
