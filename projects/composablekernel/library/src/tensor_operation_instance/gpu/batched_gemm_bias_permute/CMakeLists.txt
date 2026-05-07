# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
add_instance_library(device_batched_gemm_bias_permute_instance
    device_batched_gemm_bias_permute_m2_n3_k1_xdl_c_shuffle_f16_f16_f16_f16_instance.cpp
)

