# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_ext_bgradb "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_ext_bgradb.cpp")
target_link_libraries(sample_hipblaslt_gemm_ext_bgradb PRIVATE sample-helper)
