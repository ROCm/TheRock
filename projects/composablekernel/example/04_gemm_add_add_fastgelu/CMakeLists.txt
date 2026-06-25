# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(example_gemm_add_add_fastgelu_xdl)
add_example_executable(example_gemm_add_add_fastgelu_xdl_bf16 gemm_add_add_fastgelu_xdl_bf16.cpp)
add_example_dependencies(example_gemm_add_add_fastgelu_xdl example_gemm_add_add_fastgelu_xdl_bf16)

add_example_executable(example_gemm_add_add_fastgelu_xdl_fp16 gemm_add_add_fastgelu_xdl_fp16.cpp)
add_example_dependencies(example_gemm_add_add_fastgelu_xdl example_gemm_add_add_fastgelu_xdl_fp16)

add_example_executable(example_gemm_add_add_fastgelu_xdl_fp32 gemm_add_add_fastgelu_xdl_fp32.cpp)
add_example_dependencies(example_gemm_add_add_fastgelu_xdl example_gemm_add_add_fastgelu_xdl_fp32)

add_example_executable(example_gemm_add_add_fastgelu_xdl_int8 gemm_add_add_fastgelu_xdl_int8.cpp)
add_example_dependencies(example_gemm_add_add_fastgelu_xdl example_gemm_add_add_fastgelu_xdl_int8)

if(USE_BITINT_EXTENSION_INT4)
    add_example_executable(example_gemm_add_add_fastgelu_xdl_int4 gemm_add_add_fastgelu_xdl_int4.cpp)
    add_example_dependencies(example_gemm_add_add_fastgelu_xdl example_gemm_add_add_fastgelu_xdl_int4)
endif(USE_BITINT_EXTENSION_INT4)

list(APPEND gpu_list gfx90a gfx942 gfx950)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_example_executable(example_gemm_add_add_fastgelu_xdl_lds_direct_load_fp32 gemm_add_add_fastgelu_xdl_lds_direct_load_fp32.cpp)
        add_example_dependencies(example_gemm_add_add_fastgelu_xdl example_gemm_add_add_fastgelu_xdl_lds_direct_load_fp32)
        set(target 1)
    endif()
endforeach()
