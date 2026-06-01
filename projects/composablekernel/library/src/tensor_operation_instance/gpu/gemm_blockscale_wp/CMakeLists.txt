# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_AND_WMMA_KERNELS
if(SUPPORTED_GPU_TARGETS MATCHES "gfx9[45]|gfx12")
        set(GEMM_BLOCKSCALE_WP_INSTANCES)

        list(APPEND GEMM_BLOCKSCALE_WP_INSTANCES 
                device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_default_instance.cpp
                device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_kpadding_instance.cpp
                device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instance.cpp
                device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_kpadding_instance.cpp

                device_gemm_blockscale_wp_wmma_f8_f8_bf16/device_gemm_blockscale_wp_wmma_f8_f8_bf16_mk_nk_mn_128_128_128_comp_default_instance.cpp
                device_gemm_blockscale_wp_wmma_f8_f8_bf16/device_gemm_blockscale_wp_wmma_f8_f8_bf16_mk_nk_mn_128_128_128_mem_default_instance.cpp
                )
        check_cxx_compiler_flag("-mllvm --misched-bottomup=1" HAS_MISCHED_BOTTOMUP)
        check_cxx_compiler_flag("-mllvm --misched-prera-direction=bottomup" HAS_MISCHED_PRERA_DIRECTION)
        if(HAS_MISCHED_BOTTOMUP)
                set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--misched-bottomup=1")
                set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_kpadding_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--misched-bottomup=1")
                set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--misched-bottomup=1")
                set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_kpadding_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--misched-bottomup=1")
        elseif(HAS_MISCHED_PRERA_DIRECTION)
                set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--misched-prera-direction=bottomup")
                set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_kpadding_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--misched-prera-direction=bottomup")
                set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--misched-prera-direction=bottomup")
                set_source_files_properties(device_gemm_blockscale_wp_xdl_f8_f8_bf16/device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_kpadding_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1;-mllvm;--misched-prera-direction=bottomup")
        endif()
        add_instance_library(device_gemm_blockscale_wp_instance ${GEMM_BLOCKSCALE_WP_INSTANCES})
endif()
