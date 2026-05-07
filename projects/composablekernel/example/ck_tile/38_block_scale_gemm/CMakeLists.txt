# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx12")
    set(EXE_NAME tile_example_gemm_quant)
    add_executable(${EXE_NAME}
        gemm_quant.cpp
        gemm_abquant_quantgrouped.cpp
        gemm_aquant_quantgrouped.cpp
        gemm_aquant_quantgrouped_preshufflequant.cpp
        gemm_bquant_quantgrouped_bf8i4.cpp
        gemm_bquant_quantgrouped_fp8i4.cpp
        gemm_bquant_quantgrouped_bf16mxfp4.cpp
        gemm_bquant_quantgrouped_bf8.cpp
        gemm_bquant_quantgrouped_fp8.cpp
        gemm_bquant_quantgrouped_preshuffleb.cpp
        gemm_bquant_quantgrouped_preshufflequant.cpp
        gemm_bquant_quantgrouped_preshuffleb_preshufflequant.cpp
        gemm_quant_rowcol.cpp
        gemm_quant_tensor.cpp
        )
    target_compile_options(${EXE_NAME} PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile quant gemm tests for current target")
endif()
