# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95")
  add_executable(tile_example_grouped_gemm grouped_gemm.cpp)
  add_executable(tile_example_quant_grouped_gemm
    quant_grouped_gemm.cpp
    quant_grouped_gemm_fp8_aquant.cpp
    quant_grouped_gemm_fp8_bquant.cpp
    quant_grouped_gemm_fp8_rowcol.cpp
    quant_grouped_gemm_fp8_tensor.cpp
    quant_grouped_gemm_bf8_aquant.cpp
    quant_grouped_gemm_bf8_bquant.cpp
    quant_grouped_gemm_bf8_rowcol.cpp
    quant_grouped_gemm_bf8_tensor.cpp
  )
  add_executable(tile_example_abquant_grouped_gemm abquant_grouped_gemm.cpp)
  add_executable(tile_example_grouped_gemm_preshuffle grouped_gemm_preshuffle.cpp)
  add_executable(tile_example_grouped_gemm_multi_d grouped_gemm_multi_d.cpp)
  set(EXAMPLE_GEMM_COMPILE_OPTIONS)
  if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
  endif()
  target_compile_options(tile_example_grouped_gemm PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
  target_compile_options(tile_example_grouped_gemm_preshuffle PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
  target_compile_options(tile_example_grouped_gemm_multi_d PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
  target_compile_options(tile_example_quant_grouped_gemm PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
  target_compile_options(tile_example_abquant_grouped_gemm PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()
