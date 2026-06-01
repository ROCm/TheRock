# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_example_executable(example_convnd_fwd_xdl_fp32 convnd_fwd_xdl_fp32.cpp)
add_example_executable(example_convnd_fwd_xdl_fp16 convnd_fwd_xdl_fp16.cpp)
add_example_executable(example_convnd_fwd_xdl_bf16 convnd_fwd_xdl_bf16.cpp)
add_example_executable(example_convnd_fwd_xdl_int8 convnd_fwd_xdl_int8.cpp)
add_example_executable(example_convnd_fwd_xdl_fp8 convnd_fwd_xdl_fp8.cpp)
add_example_executable(example_convnd_fwd_xdl_bf8 convnd_fwd_xdl_bf8.cpp)
add_example_executable(example_convnd_fwd_xdl_fp16_comp_fp8 convnd_fwd_xdl_fp16_comp_fp8.cpp)
add_example_executable(example_convnd_fwd_xdl_fp8_bf8 convnd_fwd_xdl_fp8_bf8.cpp)
add_example_executable(example_convnd_fwd_xdl_bf8_fp8 convnd_fwd_xdl_bf8_fp8.cpp)
add_example_executable(example_convnd_fwd_dl_fp16 convnd_fwd_dl_fp16.cpp)
add_example_executable(example_convnd_fwd_dl_fp32 convnd_fwd_dl_fp32.cpp)
add_example_executable(example_convnd_fwd_dl_int8 convnd_fwd_dl_int8.cpp)

# only build fp64 example for the following targets
list(APPEND gpu_list gfx90a gfx942 gfx950)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_example_executable(example_convnd_fwd_xdl_fp64 convnd_fwd_xdl_fp64.cpp)
        set(target 1)
    endif()
endforeach()

list(APPEND gpu_list_tf32 gfx942 gfx950)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list_tf32 AND target EQUAL 0)
        add_example_executable(example_convnd_fwd_xdl_fp32_tf32 convnd_fwd_xdl_fp32_tf32.cpp)
        set(target 1)
    endif()
endforeach()
