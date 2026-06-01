# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_AND_WMMA_KERNELS
set(BATCHED_GEMM_INSTANCES)
list(APPEND BATCHED_GEMM_INSTANCES device_batched_gemm_wmma_universal_f16_f16_f16_gkm_gkn_gmn_instance.cpp
                                   device_batched_gemm_wmma_universal_f16_f16_f16_gkm_gnk_gmn_instance.cpp
                                   device_batched_gemm_wmma_universal_f16_f16_f16_gmk_gkn_gmn_instance.cpp
                                   device_batched_gemm_wmma_universal_f16_f16_f16_gmk_gnk_gmn_instance.cpp
                                   device_batched_gemm_wmma_universal_bf16_bf16_bf16_gkm_gkn_gmn_instance.cpp
                                   device_batched_gemm_wmma_universal_bf16_bf16_bf16_gkm_gnk_gmn_instance.cpp
                                   device_batched_gemm_wmma_universal_bf16_bf16_bf16_gmk_gkn_gmn_instance.cpp
                                   device_batched_gemm_wmma_universal_bf16_bf16_bf16_gmk_gnk_gmn_instance.cpp
                                   device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instance.cpp
                                   device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instance.cpp
                                   device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instance.cpp
                                   device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instance.cpp
                                   device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gkn_gmn_instance.cpp
                                   device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gnk_gmn_instance.cpp
                                   device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gkn_gmn_instance.cpp
                                   device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gnk_gmn_instance.cpp
                                   device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instance.cpp
                                   device_batched_gemm_xdl_f32_f32_f32_gmk_gnk_gmn_instance.cpp
                                   device_batched_gemm_xdl_f32_f32_f32_gkm_gkn_gmn_instance.cpp
                                   device_batched_gemm_xdl_f32_f32_f32_gkm_gnk_gmn_instance.cpp
                                   device_batched_gemm_xdl_int8_int8_int8_gmk_gkn_gmn_instance.cpp
                                   device_batched_gemm_xdl_int8_int8_int8_gmk_gnk_gmn_instance.cpp
                                   device_batched_gemm_xdl_int8_int8_int8_gkm_gkn_gmn_instance.cpp
                                   device_batched_gemm_xdl_int8_int8_int8_gkm_gnk_gmn_instance.cpp)
add_instance_library(device_batched_gemm_instance ${BATCHED_GEMM_INSTANCES})
