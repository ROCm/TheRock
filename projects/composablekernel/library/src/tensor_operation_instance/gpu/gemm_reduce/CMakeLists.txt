# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_AND_WMMA_KERNELS
add_instance_library(device_gemm_reduce_instance
    device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instance.cpp
    device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instance.cpp
    device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp
    device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instance.cpp

    device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_mk_kn_mn_instance.cpp
    device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_mk_nk_mn_instance.cpp
    device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp
    device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_km_nk_mn_instance.cpp
)
