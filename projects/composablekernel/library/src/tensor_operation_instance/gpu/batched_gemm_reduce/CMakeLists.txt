# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
add_instance_library(device_batched_gemm_reduce_instance
    device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn_instance.cpp
    device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn_instance.cpp
    device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn_instance.cpp
    device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn_instance.cpp
)
