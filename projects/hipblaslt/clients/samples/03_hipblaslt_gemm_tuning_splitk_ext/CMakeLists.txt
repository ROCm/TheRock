# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_tuning_splitk_ext "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_tuning_splitk_ext.cpp")
target_link_libraries(sample_hipblaslt_gemm_tuning_splitk_ext PRIVATE sample-helper)
