# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(GEMM_INSTANCES)
list(APPEND GEMM_INSTANCES device_gemm_xdl_f64_f64_f64_mk_kn_mn_instance.cpp
    device_gemm_xdl_f64_f64_f64_mk_nk_mn_instance.cpp
    device_gemm_xdl_f64_f64_f64_km_kn_mn_instance.cpp
    device_gemm_xdl_f64_f64_f64_km_nk_mn_instance.cpp
    device_gemm_xdl_f32_f32_f32_mk_kn_mn_instance.cpp
    device_gemm_xdl_f32_f32_f32_mk_nk_mn_instance.cpp
    device_gemm_xdl_f32_f32_f32_km_kn_mn_instance.cpp
    device_gemm_xdl_f32_f32_f32_km_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_f32_f32_f32_mk_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_f32_f32_f32_mk_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_f32_f32_f32_km_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_mk_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_mk_nk_mn_instance.cpp
    device_gemm_dl_f32_f32_f32_mk_kn_mn_instance.cpp
    device_gemm_dl_f32_f32_f32_mk_nk_mn_instance.cpp
    device_gemm_dl_f32_f32_f32_km_kn_mn_instance.cpp
    device_gemm_dl_f32_f32_f32_km_nk_mn_instance.cpp
    device_gemm_dl_f16_f16_f16_mk_kn_mn_instance.cpp
    device_gemm_dl_f16_f16_f16_mk_kn_mn_irregular_instance.cpp
    device_gemm_dl_f16_f16_f16_mk_nk_mn_instance.cpp
    device_gemm_dl_f16_f16_f16_mk_nk_mn_irregular_instance.cpp
    device_gemm_dl_f16_f16_f16_km_kn_mn_instance.cpp
    device_gemm_dl_f16_f16_f16_km_kn_mn_irregular_instance.cpp
    device_gemm_dl_f16_f16_f16_km_nk_mn_instance.cpp
    device_gemm_dl_f16_f16_f16_km_nk_mn_irregular_instance.cpp
    device_gemm_dpp_f16_f16_f16_km_kn_mn_instance.cpp
    device_gemm_dpp_f16_f16_f16_km_nk_mn_instance.cpp
    device_gemm_dpp_f16_f16_f16_mk_kn_mn_instance.cpp
    device_gemm_dpp_f16_f16_f16_mk_nk_mn_instance.cpp
    device_gemm_dpp_f16_f16_f16_km_kn_mn_irregular_instance.cpp
    device_gemm_dpp_f16_f16_f16_km_nk_mn_irregular_instance.cpp
    device_gemm_dpp_f16_f16_f16_mk_kn_mn_irregular_instance.cpp
    device_gemm_dpp_f16_f16_f16_mk_nk_mn_irregular_instance.cpp
    device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_lds_direct_load_f16_f16_f16_mk_nk_mn_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_kn_mn_add_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_kn_mn_default_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_kn_mn_default_pipeline_v2_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_kn_mn_default_pipeline_v2_opt_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_kn_mn_interwave_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_kn_mn_irregular_default_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_kn_mn_irregular_default_pipeline_v2_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_kn_mn_irregular_interwave_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_nk_mn_add_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_nk_mn_default_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_nk_mn_default_pipeline_v2_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_nk_mn_default_pipeline_v2_opt_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_nk_mn_interwave_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_nk_mn_irregular_default_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_nk_mn_irregular_default_pipeline_v2_instance.cpp
    device_gemm_xdl_f16_f16_f16/km_nk_mn_irregular_interwave_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_kn_mn_add_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_kn_mn_default_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_kn_mn_default_pipeline_v2_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_kn_mn_default_pipeline_v2_opt_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_kn_mn_interwave_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_kn_mn_irregular_default_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_kn_mn_irregular_default_pipeline_v2_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_kn_mn_irregular_interwave_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_nk_mn_add_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_nk_mn_default_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_nk_mn_default_pipeline_v2_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_nk_mn_default_pipeline_v2_opt_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_nk_mn_interwave_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_nk_mn_irregular_default_pipeline_v1_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_nk_mn_irregular_default_pipeline_v2_instance.cpp
    device_gemm_xdl_f16_f16_f16/mk_nk_mn_irregular_interwave_pipeline_v1_instance.cpp
    device_gemm_dl_i8_i8_i8_mk_kn_mn_instance.cpp
    device_gemm_dl_i8_i8_i8_mk_kn_mn_irregular_instance.cpp
    device_gemm_dl_i8_i8_i8_mk_nk_mn_instance.cpp
    device_gemm_dl_i8_i8_i8_mk_nk_mn_irregular_instance.cpp
    device_gemm_dl_i8_i8_i8_km_kn_mn_instance.cpp
    device_gemm_dl_i8_i8_i8_km_kn_mn_irregular_instance.cpp
    device_gemm_dl_i8_i8_i8_km_nk_mn_instance.cpp
    device_gemm_dl_i8_i8_i8_km_nk_mn_irregular_instance.cpp
    device_gemm_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_i8_i8_i8_mk_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_i8_i8_i8_km_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_i8_i8_i8_km_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_default_instance.cpp
    device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_interwave_default_instance.cpp
    device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v2_default_instance.cpp
    device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_padded_instance.cpp
    device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_interwave_padded_instance.cpp
    device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v2_padded_instance.cpp
    device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_nk_mn_instance.cpp
    device_gemm_xdl_c_shuffle_fp8_fp8_fp8_km_kn_mn_instance.cpp
    device_gemm_xdl_c_shuffle_fp8_fp8_fp8_km_nk_mn_instance.cpp
    device_gemm_wmma_f16_f16_f16_mk_kn_mn_instance.cpp
    device_gemm_wmma_f16_f16_f16_mk_nk_mn_instance.cpp
    device_gemm_wmma_f16_f16_f16_km_kn_mn_instance.cpp
    device_gemm_wmma_f16_f16_f16_km_nk_mn_instance.cpp
    device_gemm_wmma_bf16_bf16_bf16_mk_kn_mn_instance.cpp
    device_gemm_wmma_bf16_bf16_bf16_mk_nk_mn_instance.cpp
    device_gemm_wmma_bf16_bf16_bf16_km_kn_mn_instance.cpp
    device_gemm_wmma_bf16_bf16_bf16_km_nk_mn_instance.cpp
    device_gemm_wmma_int8_int8_int8_mk_kn_mn_instance.cpp
    device_gemm_wmma_int8_int8_int8_mk_nk_mn_instance.cpp
    device_gemm_wmma_int8_int8_int8_km_kn_mn_instance.cpp
    device_gemm_wmma_int8_int8_int8_km_nk_mn_instance.cpp)

add_instance_library(device_gemm_instance ${GEMM_INSTANCES})

set(ENABLE_PIPELINE_V2_OPT)

if (ENABLE_PIPELINE_V2_OPT)
    set(WAVES_PER_EU_DEFS
        CK_USE_WAVES_PER_EU=1
        CK_MIN_WAVES_PER_EU=1
        CK_MAX_WAVES_PER_EU=1
        )
    set(IGLP_OPT_DEFS
        CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT=1
        )

    # TODO: The "-vectorize-slp=false" LLVM option is a workaround to prevent inefficient instruction scheduling
    #       caused by the SLP Vectorizer. Remove this option after fix the SLP Vectorizer issue.
    # layout=NT
    set_source_files_properties(device_gemm_xdl_f16_f16_f16/km_kn_mn_default_pipeline_v2_opt_instance.cpp PROPERTIES
        COMPILE_OPTIONS ";-mllvm;-vectorize-slp=false"
        COMPILE_DEFINITIONS "${WAVES_PER_EU_DEFS};${IGLP_OPT_DEFS}")
    # layout=NN
    set_source_files_properties(device_gemm_xdl_f16_f16_f16/km_nk_mn_default_pipeline_v2_opt_instance.cpp PROPERTIES
        COMPILE_OPTIONS ";-mllvm;-vectorize-slp=false"
        COMPILE_DEFINITIONS "${WAVES_PER_EU_DEFS};${IGLP_OPT_DEFS}")
    # layout=TT
    set_source_files_properties(device_gemm_xdl_f16_f16_f16/mk_kn_mn_default_pipeline_v2_opt_instance.cpp PROPERTIES
        COMPILE_OPTIONS ";;"
        COMPILE_DEFINITIONS "${WAVES_PER_EU_DEFS};${IGLP_OPT_DEFS}")
    # layout=TN
    set_source_files_properties(device_gemm_xdl_f16_f16_f16/mk_nk_mn_default_pipeline_v2_opt_instance.cpp PROPERTIES
        COMPILE_OPTIONS ";;"
        COMPILE_DEFINITIONS "${WAVES_PER_EU_DEFS};${IGLP_OPT_DEFS}")
endif(ENABLE_PIPELINE_V2_OPT)

