# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(example_gemm_dl)

add_example_executable(example_gemm_dl_fp32 gemm_dl_fp32.cpp)
add_example_dependencies(example_gemm_dl example_gemm_dl_fp32)

add_example_executable(example_gemm_dl_fp16 gemm_dl_fp16.cpp)
add_example_dependencies(example_gemm_dl example_gemm_dl_fp16)

add_example_executable(example_gemm_dpp_fp16 gemm_dpp_fp16.cpp)

add_example_executable(example_gemm_dl_int8 gemm_dl_int8.cpp)
add_example_dependencies(example_gemm_dl example_gemm_dl_int8)
if(USE_BITINT_EXTENSION_INT4)
    add_example_executable(example_gemm_dl_int4 gemm_dl_int4.cpp)
    add_example_dependencies(example_gemm_dl example_gemm_dl_int4)
endif(USE_BITINT_EXTENSION_INT4)

add_custom_target(example_gemm_xdl)
add_example_executable(example_gemm_xdl_fp16 gemm_xdl_fp16.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16)

add_example_executable(example_gemm_xdl_fp16_v2 gemm_xdl_fp16_v2.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_v2)

add_example_executable(example_gemm_xdl_fp16_streamk_v3 gemm_xdl_fp16_streamk_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_streamk_v3)
add_example_executable(example_gemm_xdl_fp16_v3 gemm_xdl_fp16_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_v3)
add_example_executable(example_gemm_xdl_fp8_v3 gemm_xdl_fp8_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp8_v3)

add_example_executable(example_gemm_xdl_fp16_fp8_v3 gemm_xdl_fp16_fp8_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_fp8_v3)


add_example_executable(example_gemm_xdl_fp16_fp8_streamk_v3 gemm_xdl_fp16_fp8_streamk_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_fp8_streamk_v3)

add_example_executable(example_gemm_xdl_bf16_v3 gemm_xdl_bf16_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_bf16_v3)

set(GEMM_OPTIONS)
list(APPEND GEMM_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm --slp-threshold=-16")
example_compile_options(example_gemm_xdl_fp8_v3 PRIVATE ${GEMM_OPTIONS})
example_compile_options(example_gemm_xdl_bf16_v3 PRIVATE ${GEMM_OPTIONS})

list(APPEND gpu_list gfx942 gfx950 gfx1200 gfx1201 gfx12-generic)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_example_executable(example_gemm_xdl_fp16_pk_i4_v3 gemm_xdl_fp16_pk_i4_v3.cpp)
        add_example_executable(example_gemm_xdl_fp16_pk_i4_v3_b_scale gemm_xdl_fp16_pk_i4_v3_b_scale.cpp)
        add_example_executable(example_gemm_xdl_bf16_pk_i4_v3 gemm_xdl_bf16_pk_i4_v3.cpp)
        add_example_executable(example_gemm_xdl_fp8_pk_i4_bpreshuffle_v3 gemm_xdl_fp8_pk_i4_bpreshuffle_v3.cpp)
        add_example_executable(example_gemm_xdl_fp8_pk_i4_v3 gemm_xdl_fp8_pk_i4_v3.cpp)
        set(target 1)
    endif()
endforeach()

add_example_executable(example_gemm_xdl_wavelet_fp16 gemm_xdl_wavelet_fp16.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_wavelet_fp16)

add_example_executable(example_gemm_xdl_skip_b_lds_fp16 gemm_xdl_skip_b_lds_fp16.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_skip_b_lds_fp16)

add_example_executable(example_gemm_xdl_bf16 gemm_xdl_bf16.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_bf16)

add_example_executable(example_gemm_xdl_int8 gemm_xdl_int8.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_int8)

if(USE_BITINT_EXTENSION_INT4)
    add_example_executable(example_gemm_xdl_int4 gemm_xdl_int4.cpp)
    add_example_dependencies(example_gemm_xdl example_gemm_xdl_int4)
endif(USE_BITINT_EXTENSION_INT4)

add_example_executable(example_gemm_xdl_fp64 gemm_xdl_fp64.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp64)

add_example_executable(example_gemm_xdl_streamk gemm_xdl_streamk.cpp)

list(APPEND gpu_list gfx90a gfx942 gfx950)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_example_executable(example_gemm_xdl_lds_direct_load_fp32 gemm_xdl_lds_direct_load_fp32.cpp)
        add_example_dependencies(example_gemm_xdl example_gemm_xdl_lds_direct_load_fp32)

        add_example_executable(example_gemm_xdl_lds_direct_load_fp16 gemm_xdl_lds_direct_load_fp16.cpp)
        add_example_dependencies(example_gemm_xdl example_gemm_xdl_lds_direct_load_fp16)
        set(target 1)
    endif()
endforeach()

list(APPEND gpu_list gfx90a gfx942 gfx950 gfx1200 gfx1201 gfx12-generic)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_example_executable(example_gemm_xdl_bf16_streamk_v3 gemm_xdl_bf16_streamk_v3.cpp)
        add_example_dependencies(example_gemm_xdl example_gemm_xdl_bf16_streamk_v3)

        add_example_executable(example_gemm_xdl_fp8_streamk_v3 gemm_xdl_fp8_streamk_v3.cpp)
        add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp8_streamk_v3)
        set(target 1)
    endif()
endforeach()

list(APPEND gpu_list_tf32 gfx942 gfx950)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list_tf32 AND target EQUAL 0)
        add_example_executable(example_gemm_xdl_lds_direct_load_fp32_tf32 gemm_xdl_lds_direct_load_fp32_tf32.cpp)
        add_example_dependencies(example_gemm_xdl example_gemm_xdl_lds_direct_load_fp32_tf32)
        set(target 1)
    endif()
endforeach()

add_example_executable(example_gemm_xdl_fp8 gemm_xdl_fp8.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp8)

add_example_executable(example_gemm_xdl_fp8_bf8 gemm_xdl_fp8_bf8.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp8_bf8)

add_example_executable(example_gemm_xdl_fp16_fp8 gemm_xdl_fp16_fp8.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_fp8)

add_custom_target(example_gemm_wmma)
add_example_executable(example_gemm_wmma_fp16 gemm_wmma_fp16.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_fp16)
add_example_executable(example_gemm_wmma_bf16 gemm_wmma_bf16.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_bf16)
add_example_executable(example_gemm_wmma_int8 gemm_wmma_int8.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_int8)

add_example_executable(example_gemm_wmma_bf16_v3 gemm_wmma_bf16_v3.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_bf16_v3)
add_example_executable(example_gemm_wmma_bf16_pk_i4_v3 gemm_wmma_bf16_pk_i4_v3.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_bf16_pk_i4_v3)
add_example_executable(example_gemm_wmma_fp8_v3 gemm_wmma_fp8_v3.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_fp8_v3)
add_example_executable(example_gemm_wmma_fp16_v3 gemm_wmma_fp16_v3.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_fp16_v3)
add_example_executable(example_gemm_wmma_fp16_pk_i4_v3 gemm_wmma_fp16_pk_i4_v3.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_fp16_pk_i4_v3)
add_example_executable(example_gemm_wmma_fp16_fp8_v3 gemm_wmma_fp16_fp8_v3.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_fp16_fp8_v3)
add_example_executable(example_gemm_wmma_fp16_pk_i4_v3_b_scale gemm_wmma_fp16_pk_i4_v3_b_scale.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_fp16_pk_i4_v3_b_scale)
