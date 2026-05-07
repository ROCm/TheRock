# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_AND_WMMA_KERNELS
set(GEMM_UNIVERSAL_REDUCE_INSTANCES)

# XDL instances
list(APPEND GEMM_UNIVERSAL_REDUCE_INSTANCES
        device_gemm_xdl_universal_bf16_i8_bf16/device_gemm_xdl_universal_bf16_i8_bf16_mk_kn_mn_comp_default_instance.cpp
        device_gemm_xdl_universal_bf16_i8_bf16/device_gemm_xdl_universal_bf16_i8_bf16_mk_kn_mn_comp_kpadding_instance.cpp
        device_gemm_xdl_universal_bf16_i8_bf16/device_gemm_xdl_universal_bf16_i8_bf16_mk_kn_mn_comp_mnpadding_instance.cpp
        device_gemm_xdl_universal_bf16_i8_bf16/device_gemm_xdl_universal_bf16_i8_bf16_mk_kn_mn_comp_mnkpadding_instance.cpp
        device_gemm_xdl_universal_bf16_i8_bf16/device_gemm_xdl_universal_bf16_i8_bf16_mk_kn_mn_mem_v2_default_instance.cpp
        device_gemm_xdl_universal_bf16_i8_bf16/device_gemm_xdl_universal_bf16_i8_bf16_mk_kn_mn_mem_v2_kpadding_instance.cpp
        device_gemm_xdl_universal_bf16_i8_bf16/device_gemm_xdl_universal_bf16_i8_bf16_mk_kn_mn_mem_v2_mnkpadding_instance.cpp

        device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_default_instance.cpp
        device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_kpadding_instance.cpp
        device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_mnpadding_instance.cpp
        device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_mnkpadding_instance.cpp
        device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_default_instance.cpp
        device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_kpadding_instance.cpp
        device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_mnkpadding_instance.cpp

        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_default_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_kpadding_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_mnpadding_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_comp_mnkpadding_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v1_default_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v1_kpadding_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v1_mnkpadding_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v2_default_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v2_kpadding_instance.cpp
        device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn_mem_v2_mnkpadding_instance.cpp
        )

# WMMA instances
list(APPEND GEMM_UNIVERSAL_REDUCE_INSTANCES
        device_gemm_wmma_universal_bf16_bf16_bf16/device_gemm_wmma_universal_bf16_bf16_bf16_mk_kn_mn_comp_default_instance.cpp
        device_gemm_wmma_universal_bf16_i8_bf16/device_gemm_wmma_universal_bf16_i8_bf16_mk_kn_mn_comp_default_instance.cpp
        device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_mk_kn_mn_comp_default_instance.cpp
        )

add_instance_library(device_gemm_universal_reduce_instance ${GEMM_UNIVERSAL_REDUCE_INSTANCES})
