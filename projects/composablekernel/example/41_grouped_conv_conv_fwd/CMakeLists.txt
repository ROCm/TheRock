# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_example_executable(example_grouped_conv_conv_fwd_xdl_fp32 grouped_conv_conv_fwd_xdl_fp32.cpp)
add_example_executable(example_grouped_conv_conv_fwd_xdl_fp16 grouped_conv_conv_fwd_xdl_fp16.cpp)
add_example_executable(example_grouped_conv_conv_fwd_xdl_bf16 grouped_conv_conv_fwd_xdl_bf16.cpp)
if(USE_BITINT_EXTENSION_INT4)
   add_example_executable(example_grouped_conv_conv_fwd_xdl_int4 grouped_conv_conv_fwd_xdl_int4.cpp)
endif(USE_BITINT_EXTENSION_INT4)

if(NOT GPU_TARGETS MATCHES "gfx94" AND NOT GPU_TARGETS MATCHES "gfx95" AND NOT GPU_TARGETS MATCHES "gfx1")
   add_example_executable(example_grouped_conv_conv_fwd_xdl_int8 grouped_conv_conv_fwd_xdl_int8.cpp)
endif()
