# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
set(GROUPED_GEMM_FIXED_NK_INSTANCES)

list(APPEND GROUPED_GEMM_FIXED_NK_INSTANCES device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_kn_mn_instance.cpp
                                            device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_nk_mn_instance.cpp
                                            device_grouped_gemm_xdl_fixed_nk_f16_fp8_f16_mk_kn_mn_instance.cpp
                                            device_grouped_gemm_xdl_fixed_nk_f16_fp8_f16_mk_nk_mn_instance.cpp
                                            device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_kn_mn_instance.cpp
                                            device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_nk_mn_instance.cpp
                                            device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_mk_kn_mn_instance.cpp
                                            device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_mk_nk_mn_instance.cpp
                                            device_grouped_gemm_xdl_fixed_nk_bf16_bf16_bf16_mk_kn_mn_instance.cpp
                                            device_grouped_gemm_xdl_fixed_nk_bf16_bf16_bf16_mk_nk_mn_instance.cpp)

add_instance_library(device_grouped_gemm_fixed_nk_instance ${GROUPED_GEMM_FIXED_NK_INSTANCES})