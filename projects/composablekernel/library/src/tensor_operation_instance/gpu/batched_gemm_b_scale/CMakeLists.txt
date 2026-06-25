# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_AND_WMMA_KERNELS
set(BATCHED_GEMM_B_SCALE_INSTANCES)

list(APPEND BATCHED_GEMM_B_SCALE_INSTANCES 
        device_batched_gemm_b_scale_xdl_f16_i4_f16/device_batched_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_v2_default_instance.cpp
        device_batched_gemm_b_scale_wmma_f16_i4_f16/device_batched_gemm_b_scale_wmma_f16_i4_f16_mk_nk_mn_mem_default_instance.cpp
        )

set_source_files_properties(device_batched_gemm_b_scale_xdl_f16_i4_f16/device_batched_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_v2_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_batched_gemm_b_scale_wmma_f16_i4_f16/device_batched_gemm_b_scale_wmma_f16_i4_f16_mk_nk_mn_mem_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")


add_instance_library(device_batched_gemm_b_scale_instance ${BATCHED_GEMM_B_SCALE_INSTANCES})