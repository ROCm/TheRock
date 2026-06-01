# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_dgelu_bgrad_ext "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_dgelu_bgrad_ext.cpp")
target_link_libraries(sample_hipblaslt_gemm_dgelu_bgrad_ext PRIVATE sample-helper)
