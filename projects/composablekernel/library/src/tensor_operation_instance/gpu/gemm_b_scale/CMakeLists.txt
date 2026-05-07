# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_AND_WMMA_KERNELS
set(GEMM_B_SCALE_INSTANCES)

list(APPEND GEMM_B_SCALE_INSTANCES 
        device_gemm_b_scale_xdl_f16_i4_f16/device_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_v2_default_instance.cpp
        device_gemm_b_scale_wmma_f16_i4_f16/device_gemm_b_scale_wmma_f16_i4_f16_mk_nk_mn_mem_v2_default_instance.cpp
        )

set_source_files_properties(device_gemm_b_scale_xdl_f16_i4_f16/device_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_v2_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_gemm_b_scale_wmma_f16_i4_f16/device_gemm_b_scale_wmma_f16_i4_f16_mk_nk_mn_mem_v2_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")

add_instance_library(device_gemm_b_scale_instance ${GEMM_B_SCALE_INSTANCES})
