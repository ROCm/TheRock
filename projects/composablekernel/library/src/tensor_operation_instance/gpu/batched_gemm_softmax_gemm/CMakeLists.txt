# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
add_instance_library(device_batched_gemm_softmax_gemm_instance
    device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp
)
