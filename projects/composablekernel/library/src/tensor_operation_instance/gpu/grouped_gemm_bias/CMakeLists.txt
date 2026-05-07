# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
add_instance_library(device_grouped_gemm_bias_instance
   device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_kn_mn_instance.cpp
   device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_nk_mn_instance.cpp

   device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_kn_mn_instance.cpp
   device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_nk_mn_instance.cpp
)
