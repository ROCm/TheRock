# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(example_cgemm_xdl)

add_example_executable(example_cgemm_xdl_bf16 cgemm_xdl_bf16.cpp)
add_example_dependencies(example_cgemm_xdl example_cgemm_xdl_bf16)

add_example_executable(example_cgemm_xdl_fp16 cgemm_xdl_fp16.cpp)
add_example_dependencies(example_cgemm_xdl example_cgemm_xdl_fp16)

add_example_executable(example_cgemm_xdl_fp32 cgemm_xdl_fp32.cpp)
add_example_dependencies(example_cgemm_xdl example_cgemm_xdl_fp32)

add_example_executable(example_cgemm_xdl_int8 cgemm_xdl_int8.cpp)
add_example_dependencies(example_cgemm_xdl example_cgemm_xdl_int8)

if(USE_BITINT_EXTENSION_INT4)
    add_example_executable(example_cgemm_xdl_int4 cgemm_xdl_int4.cpp)
    add_example_dependencies(example_cgemm_xdl example_cgemm_xdl_int4)
endif()
