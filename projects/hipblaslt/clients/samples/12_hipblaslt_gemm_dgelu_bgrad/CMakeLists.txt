# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_dgelu_bgrad "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_dgelu_bgrad.cpp")
target_link_libraries(sample_hipblaslt_gemm_dgelu_bgrad PRIVATE sample-helper)
