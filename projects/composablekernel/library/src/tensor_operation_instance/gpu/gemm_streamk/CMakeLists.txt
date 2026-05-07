# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
add_instance_library(device_gemm_streamk_instance
   # device_gemm_xdl_streamk_f32_f32_f32_mk_kn_mn_instance.cpp
   # device_gemm_xdl_streamk_f32_f32_f32_mk_nk_mn_instance.cpp
   # device_gemm_xdl_streamk_f32_f32_f32_km_kn_mn_instance.cpp
   # device_gemm_xdl_streamk_f32_f32_f32_km_nk_mn_instance.cpp
   device_gemm_xdl_streamk_f16_f16_f16_mk_kn_mn_instance.cpp
   # device_gemm_xdl_streamk_f16_f16_f16_mk_nk_mn_instance.cpp
   # device_gemm_xdl_streamk_f16_f16_f16_km_kn_mn_instance.cpp
   # device_gemm_xdl_streamk_f16_f16_f16_km_nk_mn_instance.cpp
)
